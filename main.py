"""
OpenAI-Compatible Embedding Service
模块化结构，使用 FastAPI Router 进行蓝图管理
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from app.routes import embeddings, health, rerank
from app.services import EmbeddingService, RerankService
from app.core.config import SERVICE_HOST, SERVICE_PORT, SERVICE_WORKERS


# ===================== 生命周期管理 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理（替代 on_event）"""
    # 启动时执行
    print("\n=== 服务启动 ===")

    # 初始化 Embedding 服务
    app.state.embedding_service = EmbeddingService(model_name="bge-m3")

    # 初始化 Rerank 服务
    app.state.rerank_service = RerankService(model_name="bge-reranker-base")

    print("\n✓ 所有服务初始化完成")

    yield

    # 关闭时执行
    print("\n=== 服务关闭 ===")


# ===================== 创建应用 =====================
app = FastAPI(
    title="OpenAI-Compatible Embedding Service",
    version="3.0",
    description="简化架构的嵌入向量和重排序服务",
    lifespan=lifespan
)

# ===================== 注册路由 =====================
app.include_router(embeddings.router)
app.include_router(health.router)
app.include_router(rerank.router)


# ===================== 根路径 =====================
@app.get("/")
async def root():
    """API 根路径"""
    return {
        "service": "OpenAI-Compatible Embedding Service",
        "version": "3.1",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "rerank": "/v1/rerank",
            "batch_rerank": "/v1/rerank/batch"
        },
        "features": {
            "batch_rerank": "支持批量重排序，性能提升 2-3x"
        }
    }


# ===================== 启动服务 =====================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        workers=SERVICE_WORKERS
    )
