from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import uvicorn
import torch
from sentence_transformers import SentenceTransformer
import time

# ===================== 1. 初始化 FastAPI 应用 =====================
app = FastAPI(title="OpenAI-Compatible Embedding Service", version="1.0")

# ===================== 2. 加载 Embedding 模型（GPU加速） =====================
# 可选模型：BAAI/bge-base-zh-v1.5、moka-ai/m3e-base、BAAI/bge-m3
MODEL_NAME = "BAAI/bge-base-zh-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{DEVICE}，加载模型：{MODEL_NAME}")

# 加载模型（首次运行自动下载，后续直接加载本地文件）
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
    # 量化配置（可选，降低显存占用）
    model_kwargs={"load_in_8bit": True} if DEVICE == "cuda" else {}
)


# ===================== 3. 定义 OpenAI 标准的数据模型 =====================
class EmbeddingRequest(BaseModel):
    """对齐 OpenAI Embedding API 的请求参数"""
    input: Union[str, List[str]]  # 单文本/文本列表
    model: Optional[str] = Field(default=MODEL_NAME)  # 模型名（兼容OpenAI参数）
    encoding_format: Optional[str] = Field(default="float")  # 向量格式（float/base64）
    user: Optional[str] = None  # 用户名（兼容参数）


class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    """对齐 OpenAI Embedding API 的返回格式"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "total_tokens": 0,
            "completion_tokens": 0
        }
    )


# ===================== 4. 核心接口（对齐 OpenAI /v1/embeddings） =====================
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    try:
        # 1. 处理输入（统一转为列表）
        texts = [request.input] if isinstance(request.input, str) else request.input
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        # 2. 生成嵌入向量（GPU加速）
        start_time = time.time()
        # 归一化向量（提升检索效果）
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,  # 批量处理，提升效率
            show_progress_bar=False
        )
        print(f"嵌入耗时：{time.time() - start_time:.2f}s，处理文本数：{len(texts)}")

        # 3. 构造返回数据（严格对齐OpenAI格式）
        data = [
            EmbeddingData(index=i, embedding=embedding.tolist())
            for i, embedding in enumerate(embeddings)
        ]

        # 4. 模拟token统计（兼容OpenAI返回格式，实际可根据需求计算）
        usage = {
            "prompt_tokens": sum(len(text.split()) for text in texts),  # 简单估算token数
            "total_tokens": sum(len(text.split()) for text in texts),
            "completion_tokens": 0
        }

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=usage
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"嵌入失败：{str(e)}")


# ===================== 5. 健康检查接口 =====================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }


# ===================== 启动服务 =====================
if __name__ == "__main__":
    # 启动服务：0.0.0.0 允许外部访问，端口可自定义（如8000）
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # 单进程（避免多进程重复加载模型）
    )