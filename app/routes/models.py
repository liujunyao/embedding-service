"""
模型管理相关路由
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.config import SUPPORTED_MODELS
from app.core.model_manager import (
    MODEL_STATUS,
    MODEL_ERROR_MSG,
    load_model,
    background_load_model,
    MODEL_CACHE
)
from app.schemas import (
    ModelLoadRequest,
    ModelLoadResponse,
    ModelStatusResponse
)
import time

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models():
    """查看支持的模型列表（兼容OpenAI的模型列表接口）"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "dimension": SUPPORTED_MODELS[model_name][2]
            }
            for model_name in SUPPORTED_MODELS.keys()
        ]
    }


@router.post("/models/load", response_model=ModelLoadResponse)
async def load_models(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """加载模型（同步或异步）"""
    # 校验模型名
    invalid_models = [m for m in request.models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"无效模型名：{invalid_models}，支持的模型：{list(SUPPORTED_MODELS.keys())}"
        )

    if request.sync:
        # 同步加载
        failed_models = []
        for model_name in request.models:
            try:
                load_model(model_name)
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")

        if failed_models:
            raise HTTPException(
                status_code=500,
                detail=f"部分模型加载失败：{failed_models}"
            )
        return ModelLoadResponse(
            message="所有模型加载完成",
            loading_models=request.models
        )
    else:
        # 异步加载
        for model_name in request.models:
            background_tasks.add_task(background_load_model, model_name)
        return ModelLoadResponse(
            message="模型加载任务已提交（后台异步执行），可通过 /v1/models/status 查看进度",
            loading_models=request.models
        )


@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status(model_name: str = None):
    """查询模型状态"""
    if model_name:
        if model_name not in SUPPORTED_MODELS:
            raise HTTPException(status_code=400, detail=f"无效模型名：{model_name}")
        data = [{
            "id": model_name,
            "status": MODEL_STATUS[model_name],
            "error": MODEL_ERROR_MSG.get(model_name, None),
            "dimension": SUPPORTED_MODELS[model_name][2]
        }]
    else:
        data = [{
            "id": name,
            "status": MODEL_STATUS[name],
            "error": MODEL_ERROR_MSG.get(name, None),
            "dimension": SUPPORTED_MODELS[name][2]
        } for name in SUPPORTED_MODELS.keys()]

    return ModelStatusResponse(data=data)


@router.delete("/cache/{model_name}")
async def clear_model_cache(model_name: str = None):
    """清理指定模型缓存"""
    if model_name:
        if model_name in MODEL_CACHE:
            del MODEL_CACHE[model_name]
            MODEL_STATUS[model_name] = "unloaded"
            return {"message": f"模型 {model_name} 缓存已清理"}
        else:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不在缓存中")
    else:
        MODEL_CACHE.clear()
        for name in MODEL_STATUS.keys():
            MODEL_STATUS[name] = "unloaded"
        return {"message": "所有模型缓存已清理"}
