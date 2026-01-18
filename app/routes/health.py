"""
健康检查和系统信息路由
"""
from fastapi import APIRouter, Request
import torch

from app.core.config import SUPPORTED_MODELS, SUPPORTED_RERANKERS

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request):
    """健康检查接口"""
    # 从 app.state 获取服务实例
    embedding_service = getattr(request.app.state, "embedding_service", None)
    rerank_service = getattr(request.app.state, "rerank_service", None)

    # 获取设备信息
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 获取已加载的模型
    loaded_models = []
    if embedding_service:
        loaded_models.append(embedding_service.model_name)
    if rerank_service:
        loaded_models.append(rerank_service.model_name)

    return {
        "status": "healthy",
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "loaded_models": loaded_models,
        "supported_embedding_models": list(SUPPORTED_MODELS.keys()),
        "supported_rerank_models": list(SUPPORTED_RERANKERS.keys())
    }
