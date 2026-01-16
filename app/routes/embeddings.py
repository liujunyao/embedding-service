"""
嵌入向量生成相关路由
"""
from fastapi import APIRouter, HTTPException
import time

from app.core.config import SUPPORTED_MODELS
from app.core.model_manager import MODEL_STATUS, MODEL_ERROR_MSG, MODEL_CACHE
from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData
)

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """生成嵌入向量（仅使用已加载的模型）"""
    try:
        # 1. 处理输入
        texts = [request.input] if isinstance(request.input, str) else request.input
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        # 2. 检查模型是否已加载（不触发自动加载）
        if request.model not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型：{request.model}，支持的模型列表：{list(SUPPORTED_MODELS.keys())}"
            )

        if MODEL_STATUS[request.model] != "loaded":
            current_status = MODEL_STATUS[request.model]
            error_msg = MODEL_ERROR_MSG.get(request.model, "")

            if current_status == "loading":
                raise HTTPException(
                    status_code=503,
                    detail=f"模型 {request.model} 正在加载中，请稍后重试"
                )
            elif current_status == "failed":
                raise HTTPException(
                    status_code=500,
                    detail=f"模型 {request.model} 加载失败：{error_msg}。请使用 /v1/models/load 重新加载"
                )
            else:  # unloaded
                raise HTTPException(
                    status_code=400,
                    detail=f"模型 {request.model} 未加载，请先调用 /v1/models/load 接口加载模型"
                )

        # 3. 从缓存获取已加载的模型
        model = MODEL_CACHE[request.model]

        # 4. 生成嵌入向量
        start_time = time.time()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        print(f"嵌入耗时：{time.time() - start_time:.2f}s，处理文本数：{len(texts)}")

        # 5. 构造返回数据
        data = [
            EmbeddingData(index=i, embedding=embedding.tolist())
            for i, embedding in enumerate(embeddings)
        ]

        # 6. Token统计
        usage = {
            "prompt_tokens": sum(len(text.split()) for text in texts),
            "total_tokens": sum(len(text.split()) for text in texts),
            "completion_tokens": 0
        }

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=usage
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入失败：{str(e)}")
