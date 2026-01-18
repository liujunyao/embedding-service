"""
Embedding 服务类
遵循 KISS/YAGNI 原则:
- 应用启动时加载默认模型
- 使用线程锁保证并发安全
- 单一职责: 仅负责文本嵌入生成
"""
import torch
import threading
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from app.core.config import SUPPORTED_MODELS, ENABLE_QUANTIZATION


class EmbeddingService:
    """Embedding 服务,应用初始化时创建单例"""

    def __init__(self, model_name: str = "bge-m3"):
        """
        初始化 Embedding 服务

        Args:
            model_name: 模型名称,默认使用 bge-base-zh-v1.5
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}, "
                f"支持的模型: {list(SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_path, self.enable_quant, self.dimension = SUPPORTED_MODELS[model_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()  # 线程锁保证并发安全
        self.model = None

        # 初始化时加载模型
        self._load_model()

    def _load_model(self):
        """加载模型到内存"""
        print(f"\n=== 加载 Embedding 模型 ===")
        print(f"模型: {self.model_name}")
        print(f"路径: {self.model_path}")
        print(f"设备: {self.device}")

        model_kwargs = {"device": self.device}

        # # 量化配置
        # if self.device == "cuda" and ENABLE_QUANTIZATION and self.enable_quant:
        #     try:
        #         from transformers import BitsAndBytesConfig
        #         print(f"量化: 8-bit")
        #         quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        #         model_kwargs["model_kwargs"] = {"quantization_config": quantization_config}
        #     except ImportError as e:
        #         print(f"⚠️ 量化依赖缺失,回退到 FP16: {e}")
        # else:
        #     print(f"量化: 禁用")

        try:
            self.model = SentenceTransformer(self.model_path, **model_kwargs)
            print(f"✓ 模型加载完成 (维度: {self.dimension})")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        生成文本嵌入向量 (线程安全)

        Args:
            texts: 文本列表
            normalize: 是否归一化向量

        Returns:
            numpy.ndarray: 嵌入向量数组 shape=(len(texts), dimension)
        """
        if not texts:
            raise ValueError("输入文本不能为空")

        # 使用线程锁保证并发安全
        with self.lock:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=32,
                show_progress_bar=False
            )

        return embeddings
