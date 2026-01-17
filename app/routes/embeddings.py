"""
嵌入向量生成相关路由
"""
from fastapi import APIRouter, HTTPException, Request
import time

from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData
)

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request_data: EmbeddingRequest, request: Request):
    """生成嵌入向量"""
    try:
        # 1. 处理输入
        texts = [request_data.input] if isinstance(request_data.input, str) else request_data.input
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        # 2. 从 app.state 获取服务实例
        embedding_service = request.app.state.embedding_service

        # 3. 生成嵌入向量
        start_time = time.time()
        embeddings = embedding_service.encode(texts, normalize=True)
        print(f"嵌入耗时: {time.time() - start_time:.2f}s, 文本数: {len(texts)}")

        # 4. 构造返回数据
        data = [
            EmbeddingData(index=i, embedding=embedding.tolist())
            for i, embedding in enumerate(embeddings)
        ]

        # 5. Token统计
        usage = {
            "prompt_tokens": sum(len(text.split()) for text in texts),
            "total_tokens": sum(len(text.split()) for text in texts),
            "completion_tokens": 0
        }

        return EmbeddingResponse(
            data=data,
            model=embedding_service.model_name,
            usage=usage
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入失败: {str(e)}")
