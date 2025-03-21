# model_config.py
from peft import LoraConfig
import torch

MODEL_NAME = "bigcode/starcoder2-3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LoRA 配置参数
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# 训练参数 (部分示例，更多参数可以在 boogie_lora_finetune.py 中设置)
LEARNING_RATE = 1e-4
MAX_STEPS = 500
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
OUTPUT_DIR = "./lora-starcoder2-3b"
REPORT_TO = "tensorboard"
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
PUSH_TO_HUB = False
LABEL_NAMES = ["labels"]

#
LORA_ADAPTERS_PATH = "./lora-starcoder2-3b-adapters"
