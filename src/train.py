"""LoRA fine-tuning script for email generation."""
import os
import json
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from .config import load_config, get_device, get_lora_target_modules


def load_email_examples(data_path: str) -> list:
    """Load email examples from JSONL file."""
    examples = []
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    return examples


def format_prompt(example: dict, tokenizer=None) -> str:
    """Format email example as a training prompt."""
    base_model_name = ""
    if tokenizer:
        # Try to get model name from tokenizer
        try:
            base_model_name = tokenizer.name_or_path
        except:
            pass
    
    is_llama = "llama" in base_model_name.lower()
    
    if is_llama:
        # Use instruction format for Llama - no key_points section
        instruction = f"""Write a {example.get('tone', 'professional')} email to {example.get('recipient', 'Recipient')}.

Message: {example.get('purpose', '')}

Please write a complete, professional email that includes:
- Appropriate greeting
- Clear body paragraphs addressing the message
- Professional closing

Email:
{example.get('email', '')}"""
        
        # Use chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            # Extract instruction part (before "Email:")
            instruction_part = instruction.split("Email:")[0].strip()
            messages = [
                {"role": "system", "content": "You are a professional email writing assistant. Write clear, concise, and professional emails."},
                {"role": "user", "content": instruction_part},
                {"role": "assistant", "content": example.get('email', '')}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            prompt = instruction
    else:
        # Simple format for GPT-2 and other models - no key_points section
        prompt = f"""Write a {example.get('tone', 'professional')} email to {example.get('recipient', 'Recipient')}.

Message: {example.get('purpose', '')}

Email:
{example.get('email', '')}"""
    
    return prompt


def prepare_dataset(examples: list, tokenizer, max_length: int = 512):
    """Prepare dataset for training."""
    texts = [format_prompt(ex, tokenizer) for ex in examples]
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def train_lora_model():
    """Main training function."""
    config = load_config()
    device = get_device()
    
    print(f"Using device: {device}")
    print(f"Loading base model: {config['model']['base_model']}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    # Move to device
    if device != "cpu":
        model = model.to(device)
    
    # Prepare for LoRA - auto-detect target modules if not specified
    base_model_name = config['model']['base_model']
    target_modules = config['lora'].get('target_modules')
    if not target_modules or target_modules == []:
        target_modules = get_lora_target_modules(base_model_name)
        print(f"Auto-detected target modules for {base_model_name}: {target_modules}")
    else:
        print(f"Using configured target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=target_modules,
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load training data
    print(f"Loading training data from {config['data']['train_file']}")
    examples = load_email_examples(config['data']['train_file'])
    print(f"Loaded {len(examples)} examples")
    
    # Prepare dataset
    dataset = prepare_dataset(examples, tokenizer, config['model']['max_length'])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    # Convert learning_rate to float if it's a string (YAML sometimes reads scientific notation as string)
    learning_rate = config['training']['learning_rate']
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=learning_rate,
        warmup_steps=config['training']['warmup_steps'],
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=(device != "cpu"),  # Use FP16 for MPS/CUDA
        report_to="none",
        push_to_hub=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    train_lora_model()

