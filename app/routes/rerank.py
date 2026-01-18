"""
Rerank（重排序）相关路由
"""
from fastapi import APIRouter, HTTPException, Request
import time

from app.schemas import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankDocument,
    BatchRerankRequest,
    BatchRerankResponse,
    BatchRerankQueryResult
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


@router.post("/rerank/batch", response_model=BatchRerankResponse)
async def batch_rerank_documents(request_data: BatchRerankRequest, request: Request):
    """批量重排序文档 - 支持多个查询同时处理"""
    try:
        # 1. 从 app.state 获取服务实例
        rerank_service = request.app.state.rerank_service

        # 2. 验证请求
        if not request_data.queries:
            raise HTTPException(status_code=400, detail="查询列表不能为空")

        # 3. 准备所有查询的文档
        queries_docs = []
        all_documents = []  # 保存原始文档（用于返回）

        for query_item in request_data.queries:
            documents = []
            for doc in query_item.documents:
                if isinstance(doc, str):
                    documents.append(doc)
                elif isinstance(doc, RerankDocument):
                    documents.append(doc.text)
                else:
                    documents.append(str(doc))

            if len(documents) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"查询 '{query_item.query[:30]}...' 的文档列表不能为空"
                )

            queries_docs.append((query_item.query, documents))
            all_documents.append(documents)

        # 4. 批量计算相关性分数（高效批处理）
        start_time = time.time()
        all_scores = rerank_service.predict_batch(queries_docs)
        elapsed = time.time() - start_time

        total_docs = sum(len(docs) for _, docs in queries_docs)
        print(
            f"批量 Rerank 耗时: {elapsed:.2f}s, "
            f"查询数: {len(request_data.queries)}, "
            f"总文档数: {total_docs}, "
            f"平均: {elapsed/len(request_data.queries):.3f}s/query"
        )

        # 5. 构建每个查询的结果
        query_results = []
        total_prompt_tokens = 0
        total_tokens = 0

        for query_idx, (query_item, scores, documents) in enumerate(
            zip(request_data.queries, all_scores, all_documents)
        ):
            # 排序并构建结果
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

            # 应用 top_n 限制
            top_n = query_item.top_n
            if top_n is not None:
                scored_docs = scored_docs[:top_n]

            # 构造该查询的结果
            results = [
                RerankResult(
                    index=doc["index"],
                    relevance_score=doc["score"],
                    document=doc["document"] if request_data.return_documents else None
                )
                for doc in scored_docs
            ]

            query_results.append(
                BatchRerankQueryResult(
                    query_index=query_idx,
                    query=query_item.query,
                    results=results
                )
            )

            # 累计 token 统计
            query_tokens = len(query_item.query.split()) * len(documents)
            total_prompt_tokens += query_tokens
            total_tokens += query_tokens

        # 6. 构造返回结果
        return BatchRerankResponse(
            data=query_results,
            model=rerank_service.model_name,
            usage={
                "prompt_tokens": total_prompt_tokens,
                "total_tokens": total_tokens,
                "query_count": len(request_data.queries),
                "document_count": total_docs
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量 Rerank 失败: {str(e)}")
