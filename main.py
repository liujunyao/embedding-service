from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Literal
import uvicorn
import torch
from sentence_transformers import SentenceTransformer
import time

# ===================== 1. 初始化 FastAPI 应用 =====================
app = FastAPI(title="OpenAI-Compatible Embedding Service", version="1.0")

# ===================== 2. 模型配置（预定义支持的模型） =====================
# 可扩展：添加更多模型（如 text2vec-base-chinese、glm-4-embedding）
SUPPORTED_MODELS = {
    # 模型别名（接口中使用）: (模型实际路径/名称, 是否支持量化, 维度)
    "bge-base-zh-v1.5": ("BAAI/bge-base-zh-v1.5", True, 768),
    "bge-small-zh-v1.5": ("BAAI/bge-small-zh-v1.5", True, 384),
    "m3e-base": ("moka-ai/m3e-base", True, 768),
    "m3e-small": ("moka-ai/m3e-small", True, 384),
    "bge-m3": ("BAAI/bge-m3", True, 1024),  # bge-m3：多语言，维度1024，支持量化
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", True, 384),  # 轻量多语言，维度384
}

# 模型缓存：key=模型别名，value=SentenceTransformer实例
MODEL_CACHE: Dict[str, SentenceTransformer] = {}
# 设备配置（自动检测GPU/CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备：{DEVICE}")

ModelStatus = Literal["unloaded", "loading", "loaded", "failed"]
MODEL_STATUS: Dict[str, ModelStatus] = {name: "unloaded" for name in SUPPORTED_MODELS.keys()}
MODEL_ERROR_MSG: Dict[str, str] = {}  # 记录加载失败的错误信息

# ===================== 3. 加载模型（带缓存） =====================
def load_model(model_name: str) -> SentenceTransformer:
    """加载模型，优先从缓存获取（内部函数，带状态追踪）"""
    # 1. 检查模型是否支持
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型：{model_name}，支持的模型列表：{list(SUPPORTED_MODELS.keys())}")

    # 2. 已加载则直接返回
    if MODEL_STATUS[model_name] == "loaded":
        return MODEL_CACHE[model_name]

    # 3. 加载中则等待（避免重复加载）
    if MODEL_STATUS[model_name] == "loading":
        wait_count = 0
        while MODEL_STATUS[model_name] == "loading" and wait_count < 60:  # 最多等60秒
            time.sleep(1)
            wait_count += 1
        if MODEL_STATUS[model_name] == "loaded":
            return MODEL_CACHE[model_name]
        else:
            raise RuntimeError(f"模型 {model_name} 加载超时或失败")

    # 4. 标记为加载中
    MODEL_STATUS[model_name] = "loading"
    MODEL_ERROR_MSG.pop(model_name, None)  # 清空历史错误

    try:
        model_path, enable_quant, _ = SUPPORTED_MODELS[model_name]
        print(f"开始加载模型：{model_name}（路径：{model_path}）...")

        model_kwargs = {"device": DEVICE}
        # 量化配置（降低显存占用）
        if DEVICE == "cuda" and enable_quant:
            model_kwargs["model_kwargs"] = {"load_in_8bit": True}

        model = SentenceTransformer(
            model_path,
            **model_kwargs
        )
        # 加入缓存 + 更新状态
        MODEL_CACHE[model_name] = model
        MODEL_STATUS[model_name] = "loaded"
        print(f"模型 {model_name} 加载完成，缓存总数：{len(MODEL_CACHE)}")
        return model
    except Exception as e:
        # 更新失败状态
        MODEL_STATUS[model_name] = "failed"
        MODEL_ERROR_MSG[model_name] = str(e)
        raise RuntimeError(f"加载模型 {model_name} 失败：{str(e)}")

def background_load_model(model_name: str):
    """后台加载模型（无返回值，仅更新状态）"""
    try:
        load_model(model_name)
    except Exception as e:
        print(f"后台加载模型 {model_name} 失败：{e}")

# ===================== 3. 定义 OpenAI 标准的数据模型 =====================
class EmbeddingRequest(BaseModel):
    """对齐 OpenAI Embedding API 的请求参数"""
    input: Union[str, List[str]]  # 单文本/文本列表
    model: Optional[str] = Field(default="bge-m3")  # 模型名（兼容OpenAI参数）
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

class ModelLoadRequest(BaseModel):
    """加载模型的请求参数"""
    models: List[str] = Field(..., description="要加载的模型名列表（如 ['bge-m3', 'all-MiniLM-L6-v2']）")
    sync: bool = Field(default=False, description="是否同步加载（True=等待加载完成返回，False=后台异步加载）")

class ModelLoadResponse(BaseModel):
    """加载模型的返回结果"""
    message: str
    task_id: str = Field(default_factory=lambda: str(int(time.time())))  # 简单任务ID
    loading_models: List[str]
    status: str = "success"

class ModelStatusResponse(BaseModel):
    """模型状态查询返回"""
    object: str = "list"
    data: List[Dict[str, Union[str, Optional[str]]]]

# ===================== 4. 核心接口（对齐 OpenAI /v1/embeddings） =====================
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    try:
        # 1. 处理输入（统一转为列表）
        texts = [request.input] if isinstance(request.input, str) else request.input
        if len(texts) == 0:
            raise HTTPException(status_code=400, detail="输入文本不能为空")

        # 2. 加载指定模型（从缓存/动态加载）
        model = load_model(request.model)

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

@app.get("/v1/models")
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
                "dimension": SUPPORTED_MODELS[model_name][2]  # 向量维度
            }
            for model_name in SUPPORTED_MODELS.keys()
        ]
    }

# ===================== 5. 健康检查接口 =====================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "cached_models": list(MODEL_CACHE.keys()),  # 当前缓存的模型
        "supported_models": list(SUPPORTED_MODELS.keys())  # 支持的模型列表
    }

@app.delete("/v1/cache/{model_name}")
async def clear_model_cache(model_name: str = None):
    """清理指定模型缓存（model_name=None 清理所有）"""
    if model_name:
        if model_name in MODEL_CACHE:
            del MODEL_CACHE[model_name]
            return {"message": f"模型 {model_name} 缓存已清理"}
        else:
            raise HTTPException(status_code=404, detail=f"模型 {model_name} 不在缓存中")
    else:
        MODEL_CACHE.clear()
        return {"message": "所有模型缓存已清理"}


@app.post("/v1/models/load", response_model=ModelLoadResponse)
async def load_models(
        request: ModelLoadRequest,
        background_tasks: BackgroundTasks
):
    # 1. 校验模型名
    invalid_models = [m for m in request.models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"无效模型名：{invalid_models}，支持的模型：{list(SUPPORTED_MODELS.keys())}"
        )

    # 2. 区分同步/异步加载
    if request.sync:
        # 同步加载：等待所有模型加载完成
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
        # 异步加载：加入后台任务，立即返回
        for model_name in request.models:
            background_tasks.add_task(background_load_model, model_name)
        return ModelLoadResponse(
            message="模型加载任务已提交（后台异步执行），可通过 /v1/models/status 查看进度",
            loading_models=request.models
        )


@app.get("/v1/models/status", response_model=ModelStatusResponse)
async def get_model_status(model_name: Optional[str] = None):
    """查询模型状态（可选指定单个模型，默认查所有）"""
    if model_name:
        # 查单个模型
        if model_name not in SUPPORTED_MODELS:
            raise HTTPException(status_code=400, detail=f"无效模型名：{model_name}")
        data = [{
            "id": model_name,
            "status": MODEL_STATUS[model_name],
            "error": MODEL_ERROR_MSG.get(model_name, None),
            "dimension": SUPPORTED_MODELS[model_name][2]
        }]
    else:
        # 查所有模型
        data = [{
            "id": name,
            "status": MODEL_STATUS[name],
            "error": MODEL_ERROR_MSG.get(name, None),
            "dimension": SUPPORTED_MODELS[name][2]
        } for name in SUPPORTED_MODELS.keys()]

    return ModelStatusResponse(data=data)

# ===================== 启动服务 =====================
if __name__ == "__main__":
    # 启动服务：0.0.0.0 允许外部访问，端口可自定义（如8000）
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # 单进程（避免多进程重复加载模型）
    )