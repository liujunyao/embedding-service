"""
Rerank（重排序）相关路由
"""
from fastapi import APIRouter, HTTPException
from sentence_transformers import CrossEncoder
import time

from app.core.config import SUPPORTED_RERANKERS, ENABLE_QUANTIZATION
from app.schemas import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankDocument
)

router = APIRouter(prefix="/v1", tags=["rerank"])

# Reranker 模型缓存
RERANKER_CACHE = {}
RERANKER_STATUS = {name: "unloaded" for name in SUPPORTED_RERANKERS.keys()}


def load_reranker(model_name: str) -> CrossEncoder:
    """加载 reranker 模型"""
    if model_name not in SUPPORTED_RERANKERS:
        raise ValueError(f"不支持的 reranker 模型：{model_name}")

    # 已加载则直接返回
    if RERANKER_STATUS[model_name] == "loaded":
        return RERANKER_CACHE[model_name]

    # 加载模型
    RERANKER_STATUS[model_name] = "loading"
    try:
        import torch

        model_path, enable_quant = SUPPORTED_RERANKERS[model_name]
        print(f"开始加载 reranker 模型：{model_name}（路径：{model_path}）...")

        # 设备配置
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 构建模型参数
        model_kwargs = {"device": device}

        if device == "cuda" and ENABLE_QUANTIZATION and enable_quant:
            try:
                from transformers import BitsAndBytesConfig
                print(f"  → 启用 8-bit 量化")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["model_kwargs"] = {"quantization_config": quantization_config}
            except ImportError as e:
                print(f"  ⚠️ 量化依赖缺失，回退到 FP16: {e}")

        # 加载 CrossEncoder
        model = CrossEncoder(model_path, **model_kwargs)

        # 设置为评估模式，确保确定性推理
        if hasattr(model, 'model'):
            model.model.eval()
            # 禁用 dropout 等随机层
            torch.set_grad_enabled(False)

        RERANKER_CACHE[model_name] = model
        RERANKER_STATUS[model_name] = "loaded"
        print(f"reranker 模型 {model_name} 加载完成（设备: {device}）")
        return model

    except Exception as e:
        RERANKER_STATUS[model_name] = "failed"
        raise RuntimeError(f"加载 reranker 模型 {model_name} 失败：{str(e)}")


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """重排序文档"""
    try:
        # 1. 验证模型
        if request.model not in SUPPORTED_RERANKERS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型：{request.model}，支持的模型：{list(SUPPORTED_RERANKERS.keys())}"
            )

        # 2. 检查模型状态
        if RERANKER_STATUS[request.model] != "loaded":
            # 尝试自动加载
            try:
                load_reranker(request.model)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"模型加载失败：{str(e)}"
                )

        # 3. 获取模型
        model = RERANKER_CACHE[request.model]

        # 4. 准备文档
        documents = []
        for doc in request.documents:
            if isinstance(doc, str):
                documents.append(doc)
            elif isinstance(doc, RerankDocument):
                documents.append(doc.text)
            else:
                documents.append(str(doc))

        if len(documents) == 0:
            raise HTTPException(status_code=400, detail="文档列表不能为空")

        # 5. 构建查询-文档对
        pairs = [[request.query, doc] for doc in documents]

        # 6. 计算相关性分数（使用确定性推理）
        start_time = time.time()
        scores = model.predict(
            pairs,
            batch_size=32,  # 固定 batch size
            show_progress_bar=False,
            convert_to_numpy=True,  # 确保返回 numpy array
            num_workers=0  # 禁用多线程，确保确定性
        )
        print(f"Rerank 耗时：{time.time() - start_time:.2f}s，文档数：{len(documents)}")

        # 7. 排序并构建结果
        scored_docs = [
            {
                "index": idx,
                "score": float(score),
                "document": documents[idx] if request.return_documents else None
            }
            for idx, score in enumerate(scores)
        ]

        # 按分数降序排序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # 8. 应用 top_n 限制
        if request.top_n is not None:
            scored_docs = scored_docs[:request.top_n]

        # 9. 构造返回结果
        results = [
            RerankResult(
                index=doc["index"],
                relevance_score=doc["score"],
                document=doc["document"] if request.return_documents else None
            )
            for doc in scored_docs
        ]

        # 10. Token 统计
        usage = {
            "prompt_tokens": len(request.query.split()) * len(documents),
            "total_tokens": len(request.query.split()) * len(documents)
        }

        return RerankResponse(
            results=results,
            model=request.model,
            usage=usage
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rerank 失败：{str(e)}")


@router.get("/rerank/models")
async def list_rerank_models():
    """列出支持的 reranker 模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "reranker",
                "status": RERANKER_STATUS[model_name]
            }
            for model_name in SUPPORTED_RERANKERS.keys()
        ]
    }
