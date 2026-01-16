"""
健康检查和系统信息路由
"""
from fastapi import APIRouter
import torch

from app.core.model_manager import MODEL_CACHE, DEVICE
from app.core.config import SUPPORTED_MODELS

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cached_models": list(MODEL_CACHE.keys()),
        "supported_models": list(SUPPORTED_MODELS.keys())
    }
