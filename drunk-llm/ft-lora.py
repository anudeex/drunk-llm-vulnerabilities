import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import (
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    Trainer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,

)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import os
import wandb
from pprint import pprint

os.environ['HF_TOKEN'] = ''  # TODO: set your HF token


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


def combine_texts(examples, N=10):
    combined = []
    for i in range(0, len(examples), N):
        chunk = examples[i:i + N]
        combined.append(" ".join(chunk))  # or "\n".join(chunk)
    return combined


def generate_text(gen_pipeline, prompt, max_length=512, num_return_sequences=1):
    return gen_pipeline(prompt, max_length=max_length, num_return_sequences=num_return_sequences)[0]['generated_text']


def prepare_model(model_name, lora_config):
    # Load the pre-trained model
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply LoRA to the model
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model


def initialize_trainer(model, training_args, tokenizer, train_dataset, eval_dataset):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )


def save_lora_adapter(model, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    print(f"LoRA adapter saved at '{save_directory}'")


# Function to load a LoRA adapter
def load_lora_adapter(model_name, adapter_path, device):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    base_model.to(device)

    # Load the LoRA adapter
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    lora_model = lora_model.merge_and_unload()
    lora_model.eval()
    lora_model.to(device)

    return lora_model


def generate_and_print(pipeline, prompt, model_name):
    print(f"=== {model_name} Output ===")
    output = pipeline(prompt, max_length=512, num_return_sequences=1)[0]['generated_text']
    print(output)
    print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--hf_model_name", type=str)
    parser.add_argument("--r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--target_modules", type=str, nargs='+')

    args = parser.parse_args()
    if len(args.target_modules) == 1 and " " in args.target_modules[0]:
        args.target_modules = args.target_modules[0].split()
    pprint(args.__dict__)

    wandb.init(
        project="",  # TODO: set your project
        name=f"{args.model_name}-epochs-{args.epochs}-LoRA-finetuning",
        config=vars(args)
    )

    EPOCHS = args.epochs
    with open("./data/train-drunk-texts-data.txt", "r") as f:
        train_dataset = f.readlines()
    print("Train Dataset:", len(train_dataset))

    with open("../dataset/heldout-drunk-texts-data.txt", "r") as f:
        eval_dataset = f.readlines()
    print("Eval Dataset:", len(eval_dataset))

    hf_model_name = args.hf_model_name
    print(f"\nUsing model: {hf_model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, device_map="auto", torch_dtype=torch.float16)
    print("Tokenizer pad and eos tokens", tokenizer.pad_token, tokenizer.eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Now: Tokenizer pad and eos tokens", tokenizer.pad_token, tokenizer.eos_token)

    tokenized_train = tokenize_function(train_dataset, tokenizer, args.max_length)
    tokenized_train = Dataset.from_dict(tokenized_train)
    print(tokenized_train)

    tokenized_eval = tokenize_function(eval_dataset, tokenizer, args.max_length)
    tokenized_eval = Dataset.from_dict(tokenized_eval)
    print(tokenized_eval)

    # TODO: you might want to change as per LLM being finetuning
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,  # Target the attention modules in llama2-7b
        task_type="CAUSAL_LM"
    )

    # Prepare model
    print("Preparing model with LoRA...")
    model = prepare_model(hf_model_name, lora_config)
    print(model.print_trainable_parameters())
    print(type(model))

    # Define TrainingArguments with optimizations to reduce training time
    training_args = TrainingArguments(
        output_dir=f"./epochs-{EPOCHS}-lora-{args.model_name}",
        per_device_train_batch_size=args.batch_size,  # Increased batch size
        num_train_epochs=EPOCHS,  # Reduced number of epochs
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        gradient_accumulation_steps=2,  # Gradient accumulation to simulate larger batch size
        dataloader_num_workers=4,  # Increased number of data loading workers
        run_name=f"LoRA-Finetuning-{args.model_name}-epochs-{args.epochs}",
        report_to=["wandb"],
        logging_strategy="steps",  # Log every N steps
        save_strategy="no",
        max_grad_norm=1.0
    )

    trainer = initialize_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval
    )

    print("\n=== Starting Fine-Tuning ===\n")
    trainer.train()

    print("\n=== Saving LoRA Adapters ===\n")

    save_lora_adapter(model, f"./data/epochs-{EPOCHS}-lora-{args.model_name}-adapter")
