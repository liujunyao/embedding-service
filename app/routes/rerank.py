"""
Rerank（重排序）相关路由
"""
from fastapi import APIRouter, HTTPException, Request
import time

from app.schemas import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankDocument
)

router = APIRouter(prefix="/v1", tags=["rerank"])


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request_data: RerankRequest, request: Request):
    """重排序文档"""
    try:
        # 1. 从 app.state 获取服务实例
        rerank_service = request.app.state.rerank_service

        # 2. 准备文档
        documents = []
        for doc in request_data.documents:
            if isinstance(doc, str):
                documents.append(doc)
            elif isinstance(doc, RerankDocument):
                documents.append(doc.text)
            else:
                documents.append(str(doc))

        if len(documents) == 0:
            raise HTTPException(status_code=400, detail="文档列表不能为空")

        # 3. 计算相关性分数
        start_time = time.time()
        scores = rerank_service.predict(request_data.query, documents)
        print(f"Rerank 耗时: {time.time() - start_time:.2f}s, 文档数: {len(documents)}")

        # 4. 排序并构建结果
        scored_docs = [
            {
                "index": idx,
                "score": float(score),
                "document": documents[idx] if request_data.return_documents else None
            }
            for idx, score in enumerate(scores)
        ]

        # 按分数降序排序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # 5. 应用 top_n 限制
        if request_data.top_n is not None:
            scored_docs = scored_docs[:request_data.top_n]

        # 6. 构造返回结果
        results = [
            RerankResult(
                index=doc["index"],
                relevance_score=doc["score"],
                document=doc["document"] if request_data.return_documents else None
            )
            for doc in scored_docs
        ]

        # 7. Token 统计
        usage = {
            "prompt_tokens": len(request_data.query.split()) * len(documents),
            "total_tokens": len(request_data.query.split()) * len(documents)
        }

        return RerankResponse(
            results=results,
            model=rerank_service.model_name,
            usage=usage
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank 失败: {str(e)}")
