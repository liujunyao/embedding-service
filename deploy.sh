#!/bin/bash
set -e  # 出错时立即退出

# ===================== 配置项 =====================
VENV_PATH=".venv"
# ==================================================

# 函数：检测 CUDA 版本
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d '.' -f1-2 | tr -d '.')
        echo "Detected CUDA version: $CUDA_VERSION" >&2  # 输出到 stderr，不影响返回值
        case $CUDA_VERSION in
            129) echo "cu129" ;;
            128) echo "cu128" ;;
            124) echo "cu124" ;;
            121) echo "cu121" ;;
            118) echo "cu118" ;;
            *) echo "cpu" ;;
        esac
    else
        echo "cpu"
    fi
}

# 函数：安装 uv
install_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # 临时添加 uv 到 PATH
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
}

# 函数：创建 uv 虚拟环境
create_venv() {
    if [ -d "$VENV_PATH" ]; then
        echo "Virtual environment $VENV_PATH already exists, skip creation"
        return
    fi
    echo "Creating uv virtual environment at $VENV_PATH..."
    uv venv "$VENV_PATH"
}

# 函数：激活虚拟环境并安装依赖
install_deps() {
    # 激活虚拟环境
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"

    # 检测 CUDA 版本
    DEP_GROUP=$(check_cuda)

    # 第一步：安装基础依赖（从清华源）
    echo "Step 1: Installing base dependencies..."
    uv pip install .

    # 第二步：安装 PyTorch（从官方 wheel 源）
    echo "Step 2: Installing PyTorch for $DEP_GROUP..."
    if [ "$DEP_GROUP" = "cpu" ]; then
        uv pip install torch torchvision torchaudio numpy \
            --index-url https://download.pytorch.org/whl/$DEP_GROUP
    else
        uv pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/$DEP_GROUP
    fi
}

# 函数：验证安装
verify_install() {
    echo "Verifying installation..."
    source "$VENV_PATH/bin/activate"
    python -c "
import torch
print(f'Torch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
}

# 函数：生成启动和停止脚本
generate_scripts() {
    echo "Generating start and stop scripts..."

    # 生成启动脚本（支持前台/后台运行）
    cat > start_service.sh << 'EOF'
#!/bin/bash

VENV_PATH=".venv"
PID_FILE="service.pid"
LOG_FILE="service.log"

# 检查服务是否已运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Service is already running (PID: $PID)"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# 激活虚拟环境
source "$VENV_PATH/bin/activate"

# 判断运行模式（前台/后台）
if [ "$1" = "-d" ] || [ "$1" = "--daemon" ]; then
    echo "Starting service in background mode..."
    nohup python main.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Service started (PID: $(cat $PID_FILE))"
    echo "Log file: $LOG_FILE"
    echo "Use './stop_service.sh' to stop the service"
else
    echo "Starting service in foreground mode..."
    echo "Press Ctrl+C to stop"
    python main.py
fi
EOF
    chmod +x start_service.sh

    # 生成停止脚本
    cat > stop_service.sh << 'EOF'
#!/bin/bash

PID_FILE="service.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Service is not running (PID file not found)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Service is not running (PID $PID not found)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "Stopping service (PID: $PID)..."
kill "$PID"

# 等待进程结束（最多10秒）
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "Service stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# 如果进程仍在运行，强制终止
echo "Service did not stop gracefully, forcing termination..."
kill -9 "$PID"
rm -f "$PID_FILE"
echo "Service force stopped"
EOF
    chmod +x stop_service.sh

    # 生成状态查询脚本
    cat > status_service.sh << 'EOF'
#!/bin/bash

PID_FILE="service.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Service is not running"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "Service is running (PID: $PID)"
    echo "Memory usage:"
    ps -p "$PID" -o pid,ppid,%mem,%cpu,cmd
    exit 0
else
    echo "Service is not running (stale PID file)"
    rm -f "$PID_FILE"
    exit 1
fi
EOF
    chmod +x status_service.sh

    echo "Scripts generated successfully:"
    echo "  - start_service.sh  (use -d or --daemon for background mode)"
    echo "  - stop_service.sh   (stop the service)"
    echo "  - status_service.sh (check service status)"
}

# ===================== 主流程 =====================
echo "=== Starting deployment ==="
install_uv
create_venv
install_deps
verify_install
generate_scripts

echo -e "\n=== Deployment completed successfully ==="
echo "Virtual environment: $(realpath $VENV_PATH)"
echo ""
echo "Usage:"
echo "  Start (foreground):  ./start_service.sh"
echo "  Start (background):  ./start_service.sh -d"
echo "  Stop service:        ./stop_service.sh"
echo "  Check status:        ./status_service.sh"
echo "  View logs:           tail -f service.log"