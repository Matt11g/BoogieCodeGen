from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig # 导入 PeftModel 和 PeftConfig
import torch
from model_config import MODEL_NAME, DEVICE, LORA_ADAPTERS_PATH  # 导入配置

# 加载 Tokenizer (与 boogie_code_generator.py 相同)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 加载 *LoRA 微调后的模型* (关键修改部分)
peft_config = PeftConfig.from_pretrained(LORA_ADAPTERS_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=peft_config).to(DEVICE)
model = PeftModel.from_pretrained(model, LORA_ADAPTERS_PATH).to(DEVICE)


# 1. 定义自定义停止条件类 (CodeStopCriteria) -  与 boogie_code_generator.py 相同，复用
class CodeStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        current_token_id = input_ids[0, -1]
        current_token = self.tokenizer.decode(current_token_id)
        if "```" in current_token:
            return True
        return False

# 2. 创建函数来获取停止条件列表 (get_code_stopping_criteria) - 与 boogie_code_generator.py 相同，复用
def get_code_stopping_criteria(tokenizer):
    return StoppingCriteriaList([CodeStopCriteria(tokenizer)])

# generate_code 函数 - 与 boogie_code_generator.py 相同，复用
def generate_code(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    stopping_criteria = get_code_stopping_criteria(tokenizer)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt_for_generation = "Write a more complex Boogie procedure example:\n```boogie\n"
    generated_code = generate_code(prompt_for_generation)
    print("Generated Boogie Code (LoRA Fine-tuned Model):\n", generated_code)
