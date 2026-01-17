"""
OpenAI-Compatible Embedding Service
模块化结构，使用 FastAPI Router 进行蓝图管理
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from app.routes import models, embeddings, health, rerank
from app.core.model_manager import restore_models_on_startup, save_model_state
from app.core.config import SERVICE_HOST, SERVICE_PORT, SERVICE_WORKERS


# ===================== 生命周期管理 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理（替代 on_event）"""
    # 启动时执行
    print("\n=== 服务启动 ===")
    restore_models_on_startup()
    yield
    # 关闭时执行
    print("\n=== 服务关闭 ===")
    save_model_state()


# ===================== 创建应用 =====================
app = FastAPI(
    title="OpenAI-Compatible Embedding Service",
    version="2.0",
    description="模块化嵌入向量生成服务，支持多模型管理",
    lifespan=lifespan
)

# ===================== 注册路由 =====================
app.include_router(embeddings.router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(rerank.router)


# ===================== 根路径 =====================
@app.get("/")
async def root():
    """API 根路径"""
    return {
        "service": "OpenAI-Compatible Embedding Service",
        "version": "2.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "model_status": "/v1/models/status",
            "load_models": "/v1/models/load",
            "rerank": "/v1/rerank",
            "rerank_models": "/v1/rerank/models"
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
