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

        # 1. 构建instruction 部分；拼接 instructions 和 outputs 列表，并添加 EOS token
        prompt_texts = ["instruction: " + instruction + " output: " for instruction in instructions]
        full_texts = [prompt + output + tokenizer.eos_token for prompt, output in zip(prompt_texts, outputs)]

        # 2. Tokenize 完整的文本 (应用 padding 和 truncation)；Tokenize 仅包含提示（指令）的部分 (不 padding, 不 truncation)
        tokenized_full = tokenizer(
            full_texts, 
            truncation=True, 
            padding="max_length", # 或者 'longest', 或者 False 如果用 DataCollator
            max_length=512,
            return_tensors=None, # 返回 list，方便后面处理
            add_special_tokens=False # eos_token 已手动添加，这里避免重复添加 (如果适用)
        )

        tokenized_prompts = tokenizer(
            prompt_texts,
            truncation=False, # 不需要截断
            padding=False,    # 不需要填充
            return_tensors=None, # 返回 list
            add_special_tokens=False # 同上
        )

        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']
        labels = [list(ids) for ids in input_ids] # 转为 list of lists 方便修改

        # 3. 根据提示部分的实际长度来屏蔽 labels
        for i in range(len(labels)):
            prompt_len = len(tokenized_prompts['input_ids'][i])
            # 确保屏蔽长度不超过 labels 的实际长度 (考虑 truncation)
            actual_prompt_len_in_labels = min(prompt_len, len(labels[i]))
            for j in range(actual_prompt_len_in_labels):
                labels[i][j] = -100

            # 可选：如果 output 部分也被截断了，可以考虑将末尾的 padding token 也设为 -100
            # (如果使用了 DataCollatorForLanguageModeling，它通常会自动处理 padding 部分的 label)
            # 如果在这里手动 padding='max_length'，需要找到 padding 的起始位置
            # try:
            #     pad_start_index = attention_mask[i].index(0) # 找到第一个0 (padding token)
            #     for k in range(pad_start_index, len(labels[i])):
            #          labels[i][k] = -100
            # except ValueError: # 没有 padding token
            #     pass


        # 将 list 转回 tensor (如果前面没用 return_tensors="pt")
        input_ids_tensor = torch.tensor(input_ids)
        attention_mask_tensor = torch.tensor(attention_mask)
        labels_tensor = torch.tensor(labels)


        return {
            # 返回 Tensor
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor
        }

    # 在 map 时，可能需要设置 batched=True, batch_size=合理值
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True, # 批量处理效率更高
        # batch_size=1000, # 可以指定批大小
        remove_columns=train_dataset.column_names # 移除原始列
    )
    # tokenized_dataset.set_format("torch") # 不再需要，因为 tokenize_function 已返回 tensor
    train_dataset = tokenized_dataset

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
    print(f"LoRA 微调完成！LoRA adapters 已保存到 {LORA_ADAPTERS_PATH} 目录。")


if __name__ == "__main__":
    finetune_lora()
