#!/bin/bash
set -e  # 出错时立即退出

# ===================== 配置项 =====================
VENV_PATH=".venv"
TORCH_VERSION="2.2.0"
# ==================================================

# 函数：检测 CUDA 版本
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d '.' -f1-2 | tr -d '.')
        echo "Detected CUDA version: $CUDA_VERSION"
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

    # 检测 CUDA 并安装 torch
    DEP_GROUP=$(check_cuda)
    echo "Installing torch $TORCH_VERSION for $DEP_GROUP..."
    uv pip install "torch==$TORCH_VERSION" "torchvision==0.17.0" "torchaudio==2.2.0" \
        --index-url https://download.pytorch.org/whl/cpu

    # 安装其他依赖
    echo "Installing other dependencies..."
    uv pip install fastapi>=0.104.0 uvicorn>=0.24.0 sentence-transformers>=2.7.0 pydantic>=2.5.0
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

# 函数：生成启动脚本
generate_start_script() {
    echo "Generating start script..."
    cat > start_service.sh << EOF
#!/bin/bash
source "$VENV_PATH/bin/activate"
python main.py
EOF
    chmod +x start_service.sh
    echo "Start script generated: ./start_service.sh"
}

# ===================== 主流程 =====================
echo "=== Starting deployment ==="
install_uv
create_venv
install_deps
verify_install
generate_start_script

echo -e "\n=== Deployment completed successfully ==="
echo "Virtual environment: $(realpath $VENV_PATH)"
echo "To start service:"
echo "  1. Manual: source $VENV_PATH/bin/activate && python embedding_service.py"
echo "  2. Auto: ./start_service.sh"