"""
配置管理模块
"""
from typing import Dict, Tuple
from pathlib import Path

# ===================== 路径配置 =====================
# 项目根目录（main.py 所在目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ===================== 模型配置 =====================
SUPPORTED_MODELS: Dict[str, Tuple[str, bool, int]] = {
    # 模型别名: (Hugging Face 路径, 是否支持量化, 向量维度)
    "bge-base-zh-v1.5": ("BAAI/bge-base-zh-v1.5", True, 768),
    "bge-small-zh-v1.5": ("BAAI/bge-small-zh-v1.5", True, 384),
    "m3e-base": ("moka-ai/m3e-base", True, 768),
    "m3e-small": ("moka-ai/m3e-small", True, 384),
    "bge-m3": ("BAAI/bge-m3", True, 1024),
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", True, 384),
}

# ===================== 量化配置 =====================
# 设置为 False 禁用量化（使用 FP16），True 启用 8-bit 量化
ENABLE_QUANTIZATION = True

# ===================== 持久化配置 =====================
MODEL_STATE_FILE = str(PROJECT_ROOT / "model_state.json")  # 使用绝对路径

# ===================== 服务配置 =====================
SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8000
SERVICE_WORKERS = 1
