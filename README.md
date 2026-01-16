# Embedding Service

OpenAI 兼容的嵌入向量生成服务，支持多模型管理和自动持久化。

## 快速开始

### 1. 部署服务

```bash
# Linux/macOS
bash deploy.sh

# Windows
deploy.bat
```

### 2. 启动服务

```bash
# 前台运行
./start_service.sh

# 后台运行
./start_service.sh -d
```

### 3. 停止服务

```bash
./stop_service.sh
```

## 主要功能

✅ **多模型支持** - bge-m3, all-MiniLM-L6-v2 等多个模型
✅ **自动持久化** - 服务重启后自动恢复已加载的模型
✅ **模块化架构** - 清晰的代码组织，易于维护和扩展
✅ **OpenAI 兼容** - 兼容 OpenAI Embedding API 格式
✅ **量化支持** - 可选 8-bit 量化，降低显存占用
✅ **跨平台** - 支持 Linux、macOS、Windows

## API 使用

### 加载模型

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"models": ["bge-m3"], "sync": true}'
```

### 生成嵌入向量

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "Hello World"}'
```

### 查询模型状态

```bash
curl http://localhost:8000/v1/models/status
```

## 项目结构

```
embedding-service/
├── app/                    # 应用代码
│   ├── core/              # 核心模块（配置、模型管理）
│   ├── routes/            # 路由蓝图
│   └── schemas/           # 数据模型
├── main.py                # 入口文件
├── deploy.sh              # 部署脚本
└── pyproject.toml         # 项目配置
```

## 配置

### 修改支持的模型

编辑 `app/core/config.py`:

```python
SUPPORTED_MODELS = {
    "bge-m3": ("BAAI/bge-m3", True, 1024),
    # 添加更多模型...
}
```

### 启用量化

编辑 `app/core/config.py`:

```python
ENABLE_QUANTIZATION = True  # 启用 8-bit 量化
```

需要安装量化依赖：

```bash
source .venv/bin/activate
uv pip install .[quantization]
```

## 技术栈

- **FastAPI** - Web 框架
- **Sentence-Transformers** - 嵌入模型
- **PyTorch** - 深度学习框架
- **uv** - 包管理工具

## License

MIT
