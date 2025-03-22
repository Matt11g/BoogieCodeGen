from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model_config import MODEL_NAME, DEVICE

# 加载 Tokenizer 和 模型 (在函数外部加载，只加载一次)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)


def generate_code(instruction, max_length=200):
    prompt = f"instruction:{instruction} output:\n```boogie"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    user_instruction = "write a Boogie procedure to calculate the factorial of a number\n"
    generated_code = generate_code(user_instruction)
    print("Generated Boogie Code:\n", generated_code)
