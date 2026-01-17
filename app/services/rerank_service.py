"""
Rerank 服务类
遵循 KISS/YAGNI 原则:
- 应用启动时加载默认模型
- 使用线程锁保证并发安全
- 单一职责: 仅负责文档重排序
"""
import torch
import threading
from sentence_transformers import CrossEncoder
from typing import List
import numpy as np

from app.core.config import SUPPORTED_RERANKERS, ENABLE_QUANTIZATION


class RerankService:
    """Rerank 服务,应用初始化时创建单例"""

    def __init__(self, model_name: str = "bge-reranker-base"):
        """
        初始化 Rerank 服务

        Args:
            model_name: 模型名称,默认使用 bge-reranker-base
        """
        if model_name not in SUPPORTED_RERANKERS:
            raise ValueError(
                f"不支持的模型: {model_name}, "
                f"支持的模型: {list(SUPPORTED_RERANKERS.keys())}"
            )

        self.model_name = model_name
        self.model_path, self.enable_quant = SUPPORTED_RERANKERS[model_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()  # 线程锁保证并发安全
        self.model = None

        # 初始化时加载模型
        self._load_model()

    def _load_model(self):
        """加载模型到内存"""
        print(f"\n=== 加载 Rerank 模型 ===")
        print(f"模型: {self.model_name}")
        print(f"路径: {self.model_path}")
        print(f"设备: {self.device}")

        model_kwargs = {"device": self.device}

        # 量化配置
        if self.device == "cuda" and ENABLE_QUANTIZATION and self.enable_quant:
            try:
                from transformers import BitsAndBytesConfig
                print(f"量化: 8-bit")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["model_kwargs"] = {"quantization_config": quantization_config}
            except ImportError as e:
                print(f"⚠️ 量化依赖缺失,回退到 FP16: {e}")
        else:
            print(f"量化: 禁用")

        try:
            self.model = CrossEncoder(self.model_path, **model_kwargs)

            # 设置为评估模式,确保确定性推理
            if hasattr(self.model, "model"):
                self.model.model.eval()
                torch.set_grad_enabled(False)

            print(f"✓ 模型加载完成")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def predict(self, query: str, documents: List[str]) -> np.ndarray:
        """
        计算查询与文档的相关性分数 (线程安全)

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            numpy.ndarray: 相关性分数数组 shape=(len(documents),)
        """
        if not documents:
            raise ValueError("文档列表不能为空")

        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]

        # 使用线程锁保证并发安全
        with self.lock:
            scores = self.model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                num_workers=0  # 禁用多线程,确保确定性
            )

        return scores
