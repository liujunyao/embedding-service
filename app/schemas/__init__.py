"""
Pydantic 数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Literal
import time

# ===================== Embedding 相关模型 =====================
class EmbeddingRequest(BaseModel):
    """对齐 OpenAI Embedding API 的请求参数"""
    input: Union[str, List[str]]  # 单文本/文本列表
    model: Optional[str] = Field(default="bge-m3")  # 模型名
    encoding_format: Optional[str] = Field(default="float")  # 向量格式
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


# ===================== 模型管理相关模型 =====================
class ModelLoadRequest(BaseModel):
    """加载模型的请求参数"""
    models: List[str] = Field(..., description="要加载的模型名列表")
    sync: bool = Field(default=False, description="是否同步加载")


class ModelLoadResponse(BaseModel):
    """加载模型的返回结果"""
    message: str
    task_id: str = Field(default_factory=lambda: str(int(time.time())))
    loading_models: List[str]
    status: str = "success"


class ModelStatusResponse(BaseModel):
    """模型状态查询返回"""
    object: str = "list"
    data: List[Dict[str, Union[str, int, None]]]


ModelStatus = Literal["unloaded", "loading", "loaded", "failed"]


# ===================== Rerank 相关模型 =====================
class RerankDocument(BaseModel):
    """Rerank 文档"""
    text: str = Field(..., description="文档内容")
    metadata: Optional[Dict] = Field(default=None, description="文档元数据")


class RerankRequest(BaseModel):
    """Rerank 请求参数"""
    model: str = Field(default="bge-reranker-base", description="Reranker 模型名")
    query: str = Field(..., description="查询文本")
    documents: List[Union[str, RerankDocument]] = Field(..., description="待排序的文档列表")
    top_n: Optional[int] = Field(default=None, description="返回前 N 个结果")
    return_documents: bool = Field(default=True, description="是否返回文档内容")


class RerankResult(BaseModel):
    """单个 Rerank 结果"""
    index: int = Field(..., description="文档在原始列表中的索引")
    relevance_score: float = Field(..., description="相关性分数")
    document: Optional[Union[str, Dict]] = Field(default=None, description="文档内容")


class RerankResponse(BaseModel):
    """Rerank 响应"""
    object: str = "list"
    results: List[RerankResult]
    model: str
    usage: dict = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    )

