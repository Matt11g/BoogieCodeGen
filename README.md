# Boogie Code Generation with Large Language Model (LoRA Fine-tuning)

This project demonstrates how to use a Large Language Model (specifically StarCoder2-3B) to generate Boogie code, and how to fine-tune the model using LoRA (Low-Rank Adaptation) for improved Boogie code generation capabilities.

## File Structure

*   `boogie_code_generator.py`: Contains the `generate_code` function for generating Boogie code using a pre-trained or fine-tuned model.
*   `boogie_finetuned_code_generator.py`: Contains the `generate_code` function for generating Boogie code using the **LoRA fine-tuned StarCoder2-3B model**. This script loads and utilizes the fine-tuned LoRA adapters.
*   `boogie_lora_finetune.py`: Contains the code for LoRA fine-tuning of StarCoder2-3B on a Boogie code dataset.
*   `model_config.py`: Contains configuration parameters for the model, LoRA, and training.
*   `requirements.txt`: Lists the Python library dependencies for the project.
*   `README.md`: This file, providing a project overview.

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run LoRA Fine-tuning

To fine-tune LLM with LoRA on Boogie code data, run:

```
python boogie_lora_finetune.py
```

### 3. Generate Boogie Code

You have two options for generating Boogie code:

#### a) Generate Boogie Code with the Pre-trained Model (Baseline)

To generate Boogie code using the original, pre-trained StarCoder2-3B model (without fine-tuning), run:

```
from boogie_code_generator import generate_code

prompt = "Write a Boogie procedure to calculate the factorial of a number:\n```boogie\n"
generated_code = generate_code(prompt)
print(generated_code)
```

Or, you can directly run boogie_code_generator.py for a quick example:

```
python boogie_code_generator.py
```

#### b) Generate Boogie Code with the LoRA Fine-tuned Model

To generate Boogie code using the LoRA fine-tuned StarCoder2-3B model, run:

```
from boogie_finetuned_code_generator import generate_code

prompt = "Write a more complex Boogie procedure example:\n```boogie\n"
generated_code = generate_code(prompt)
print(generated_code)
```

Or, you can directly run boogie_finetuned_code_generator.py for a quick example:

```
python boogie_finetuned_code_generator.py
```

## Configuration

Model name, LoRA parameters, training arguments, and other configurations are defined in model_config.py. Modify this file to adjust the project settings.
