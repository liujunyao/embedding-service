"""
模型加载和管理模块
"""
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict
import time
import json
import os
from pathlib import Path

from app.core.config import SUPPORTED_MODELS, ENABLE_QUANTIZATION, MODEL_STATE_FILE
from app.schemas import ModelStatus

# ===================== 全局状态 =====================
MODEL_CACHE: Dict[str, SentenceTransformer] = {}
MODEL_STATUS: Dict[str, ModelStatus] = {name: "unloaded" for name in SUPPORTED_MODELS.keys()}
MODEL_ERROR_MSG: Dict[str, str] = {}

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"使用设备：{DEVICE}")
if DEVICE == "cuda":
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU 显存: {total_memory:.2f}GB")
    print(f"量化配置: {'已启用 (8-bit)' if ENABLE_QUANTIZATION else '已禁用 (FP16)'}")


# ===================== 持久化工具函数 =====================
def save_model_state():
    """保存已加载模型列表到文件"""
    loaded_models = [
        name for name, status in MODEL_STATUS.items()
        if status == "loaded"
    ]
    try:
        print(f"准备保存模型状态到: {MODEL_STATE_FILE}")
        print(f"已加载的模型: {loaded_models}")
        print(f"当前 MODEL_STATUS: {dict(MODEL_STATUS)}")

        with open(MODEL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "loaded_models": loaded_models,
                "timestamp": int(time.time())
            }, f, indent=2)

        # 验证写入
        if os.path.exists(MODEL_STATE_FILE):
            with open(MODEL_STATE_FILE, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                print(f"✓ 已保存模型状态到 {MODEL_STATE_FILE}: {saved_data['loaded_models']}")
        else:
            print(f"⚠️ 文件写入后不存在: {MODEL_STATE_FILE}")

    except Exception as e:
        print(f"❌ 保存模型状态失败: {e}")
        import traceback
        traceback.print_exc()


def load_model_state() -> list:
    """从文件恢复已加载模型列表"""
    if not os.path.exists(MODEL_STATE_FILE):
        print(f"模型状态文件不存在: {MODEL_STATE_FILE}")
        return []

    try:
        with open(MODEL_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            loaded_models = data.get("loaded_models", [])
            timestamp = data.get("timestamp", 0)

            valid_models = [m for m in loaded_models if m in SUPPORTED_MODELS]
            if valid_models:
                print(f"从 {MODEL_STATE_FILE} 恢复模型列表: {valid_models}")
                print(f"  上次保存时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
            return valid_models
    except Exception as e:
        print(f"加载模型状态失败: {e}")
        return []


def check_model_cache_exists(model_name: str) -> bool:
    """检查模型缓存文件是否已存在"""
    if model_name not in SUPPORTED_MODELS:
        return False

    model_path = SUPPORTED_MODELS[model_name][0]

    # 尝试多种可能的缓存路径格式
    cache_base = Path.home() / ".cache"
    possible_paths = [
        # Sentence-Transformers 默认路径格式
        cache_base / "torch" / "sentence_transformers" / model_path.replace("/", "_"),
        cache_base / "huggingface" / "hub" / f"models--{model_path.replace('/', '--')}",
        # Transformers 默认路径格式
        cache_base / "huggingface" / "transformers" / model_path.replace("/", "--"),
    ]

    for cache_dir in possible_paths:
        if cache_dir.exists() and any(cache_dir.iterdir()):
            print(f"  ✓ 检测到本地缓存: {cache_dir}")
            return True

    # 如果所有路径都不存在，打印调试信息
    print(f"  ✗ 模型 {model_name} 本地缓存不存在")
    print(f"    已检查路径:")
    for path in possible_paths:
        print(f"      - {path} {'(存在但为空)' if path.exists() else '(不存在)'}")

    return False


# ===================== 模型加载函数 =====================
def load_model(model_name: str) -> SentenceTransformer:
    """加载模型，优先从缓存获取"""
    # 1. 检查模型是否支持
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型：{model_name}，支持的模型列表：{list(SUPPORTED_MODELS.keys())}")

    # 2. 已加载则直接返回
    if MODEL_STATUS[model_name] == "loaded":
        return MODEL_CACHE[model_name]

    # 3. 加载中则等待
    if MODEL_STATUS[model_name] == "loading":
        wait_count = 0
        while MODEL_STATUS[model_name] == "loading" and wait_count < 60:
            time.sleep(1)
            wait_count += 1
        if MODEL_STATUS[model_name] == "loaded":
            return MODEL_CACHE[model_name]
        else:
            raise RuntimeError(f"模型 {model_name} 加载超时或失败")

    # 4. 标记为加载中
    MODEL_STATUS[model_name] = "loading"
    MODEL_ERROR_MSG.pop(model_name, None)

    try:
        model_path, enable_quant, _ = SUPPORTED_MODELS[model_name]
        print(f"开始加载模型：{model_name}（路径：{model_path}）...")

        model_kwargs = {"device": DEVICE}

        # 量化配置
        if DEVICE == "cuda" and ENABLE_QUANTIZATION and enable_quant:
            try:
                from transformers import BitsAndBytesConfig

                print(f"  → 启用 8-bit 量化")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["model_kwargs"] = {"quantization_config": quantization_config}
            except ImportError as e:
                print(f"  ⚠️ 量化依赖缺失，回退到 FP16: {e}")
                model_kwargs["device"] = DEVICE

        model = SentenceTransformer(model_path, **model_kwargs)

        # 更新状态
        MODEL_CACHE[model_name] = model
        MODEL_STATUS[model_name] = "loaded"
        print(f"模型 {model_name} 加载完成，缓存总数：{len(MODEL_CACHE)}")

        # 持久化
        save_model_state()

        return model
    except Exception as e:
        MODEL_STATUS[model_name] = "failed"
        MODEL_ERROR_MSG[model_name] = str(e)
        raise RuntimeError(f"加载模型 {model_name} 失败：{str(e)}")


def background_load_model(model_name: str):
    """后台加载模型"""
    try:
        load_model(model_name)
    except Exception as e:
        print(f"后台加载模型 {model_name} 失败：{e}")


# ===================== 启动时恢复模型 =====================
def restore_models_on_startup():
    """服务启动时恢复之前加载的模型"""
    print("\n=== 服务启动，检查是否需要恢复模型 ===")

    previous_models = load_model_state()
    if not previous_models:
        print("未发现之前加载的模型，跳过恢复")
        return

    models_to_load = []
    for model_name in previous_models:
        if check_model_cache_exists(model_name):
            models_to_load.append(model_name)
        else:
            print(f"  ✗ 模型 {model_name} 本地缓存不存在，跳过")

    if not models_to_load:
        print("所有之前加载的模型都没有本地缓存，跳过恢复")
        return

    print(f"将在后台恢复加载以下模型: {models_to_load}")

    from concurrent.futures import ThreadPoolExecutor

    def load_models_in_background():
        for name in models_to_load:
            try:
                print(f"  → 恢复加载模型: {name}")
                load_model(name)
                print(f"  ✓ 模型 {name} 恢复完成")
            except Exception as e:
                print(f"  ✗ 模型 {name} 恢复失败: {e}")

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(load_models_in_background)

    print("模型恢复任务已提交到后台执行")
