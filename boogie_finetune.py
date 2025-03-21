# boogie_lora_finetune.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from model_config import MODEL_NAME, DEVICE, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LEARNING_RATE, MAX_STEPS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, OUTPUT_DIR, REPORT_TO, SAVE_STEPS, SAVE_TOTAL_LIMIT, PUSH_TO_HUB, LABEL_NAMES, LORA_ADAPTERS_PATH  # 导入配置

# 加载 Tokenizer (只加载 tokenizer，模型在 generate 中加载或从外部传入)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def finetune_lora():
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 将 LoRA 应用于模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 准备训练数据 (示例，你需要替换成你自己的实际数据集)
    train_dataset_text = [
        """procedure Add(x: int, y: int) returns (result: int)
        requires true;
        ensures result == x + y;
        {
            result := x + y;
        }""",
        """procedure Subtract(x: int, y: int) returns (result: int)
        requires true;
        ensures result == x - y;
        {
            result := x - y;
        }""",
        # ... (更多 Boogie procedure 示例)
    ]

    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        return {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask
        }

    tokenized_dataset = tokenize_function(train_dataset_text)
    train_dataset = Dataset.from_dict(tokenized_dataset)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        push_to_hub=PUSH_TO_HUB,
        report_to=REPORT_TO,
        label_names=LABEL_NAMES
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                    'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                    'labels': torch.stack([f['input_ids'] for f in data])},
        label_names=LABEL_NAMES
    )

    # 开始训练
    trainer.train()

    # 保存 LoRA adapters
    model.save_pretrained(LORA_ADAPTERS_PATH)
    print("LoRA 微调完成！LoRA adapters 已保存到 ./lora-starcoder2-3b-adapters 目录。")


if __name__ == "__main__":
    finetune_lora()
