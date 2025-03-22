# boogie_lora_finetune.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset, load_dataset
from model_config import MODEL_NAME, DEVICE, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, LEARNING_RATE, MAX_STEPS, PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, OUTPUT_DIR, REPORT_TO, SAVE_STEPS, SAVE_TOTAL_LIMIT, PUSH_TO_HUB, LORA_ADAPTERS_PATH  # 导入配置

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def finetune_lora():
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # 准备 SFT 训练数据 (修改部分)
    dataset_path = "boogie_sft_dataset.jsonl"
    train_dataset = load_dataset("json", data_files=dataset_path)["train"]

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

    def tokenize_function(examples):
        instructions = examples["instruction"] #  提取 instruction 字段
        outputs = examples["output"] # 提取 output 字段
        model_inputs = []
        labels_list = []
        for instruction, output in zip(instructions, outputs):
            # 1. 拼接 instruction 和 output，并添加 EOS token (重要!)
            full_text = "instruction: " + instruction + " output: " + output + tokenizer.eos_token
            tokenized_full_text = tokenizer(full_text, truncation=True, padding="longest", return_tensors="pt")

            input_ids = tokenized_full_text.input_ids
            attention_mask = tokenized_full_text.attention_mask
            labels = input_ids.clone() # 初始 labels 和 input_ids 相同

            # 2. 找到 output 部分的起始位置 (token 级别)
            instruction_part = "instruction: " + instruction + " output: "
            tokenized_instruction_part = tokenizer(instruction_part, truncation=True, return_tensors="pt")
            instruction_len = tokenized_instruction_part.input_ids.shape[1]

            # 3. 将 instruction 部分的 labels 屏蔽为 -100
            labels[:, :instruction_len] = -100

            model_inputs.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            labels_list.append(labels)
        
        #  将列表堆叠成 tensor (batched=True 情况下会一次处理多个 examples)
        batch_input_ids = torch.cat([item["input_ids"] for item in model_inputs])
        batch_attention_mask = torch.cat([item["attention_mask"] for item in model_inputs])
        batch_labels = torch.cat(labels_list)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels  # 返回屏蔽后的 labels
        }

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) #  现在包含 labels 列
    train_dataset = tokenized_dataset #  更新 train_dataset 变量

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
        label_names=["labels"]
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        #data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                    #'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                    #'labels': torch.stack([f['labels'] for f in data])}
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # 或者使用这个更通用的 data collator，mlm=False 因为我们不是 masked language modeling
    )

    # 开始训练
    trainer.train()

    # 保存 LoRA adapters
    model.save_pretrained(LORA_ADAPTERS_PATH)
    print("LoRA 微调完成！LoRA adapters 已保存到 ./lora-starcoder2-3b-adapters 目录。")


if __name__ == "__main__":
    finetune_lora()
