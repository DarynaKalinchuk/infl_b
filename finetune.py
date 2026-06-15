from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
import argparse
import warnings
import os
import torch
import random
import numpy as np
from huggingface_hub import login
warnings.filterwarnings("ignore")
import sys

seed = 1
print(f"Setting random seed: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


with open("settings_txt/TOKENS.txt", "r") as f:
    line = f.read().strip()

login(token=line.split("=", 1)[1].strip().strip('"'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Olmo', help='model name')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='whether to quantize the LLM')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--logging_step', type=int, default=10, help='logging step')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lora_r', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    args = parser.parse_args()
    
    os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

    with open("settings_txt/target_modules.txt") as f:
        target_modules = [
            line.strip()
            for line in f
            if line.strip()
        ]


    MODELS = {}
    with open("settings_txt/models.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            MODELS[key] = value

    
    if args.model in MODELS.keys():
        model_name = MODELS[args.model]
    else:
        raise ValueError("Invalid model name")
    
    model_name, chat_template = template_setting(args.model)

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    dataset = load_from_disk("datasets/" + args.dataset)
    train_dataset = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=args.max_length)  

    print(f"Training {args.model} for {args.epochs} epochs with batch size {args.batch_size}")


    save_path = f"lora_adapter/{args.model}/{args.dataset}_{args.epochs}"


    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map='auto'
    )
    model.config.use_cache = False

    print(f"Model {model_name} loaded successfully.")

    for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        os.environ.pop(var, None)

    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_step,
        save_steps=10,
        save_total_limit=10, # number of checkpoints
        remove_unused_columns=False,
        learning_rate = 5e-5
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )



    model = get_peft_model(model, lora_config)

    if args.model == "randomOlmo":
        model.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")
        sys.exit()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset
    )
    
    trainer.train()
    
    
    print("Training completed.")
    trainer.save_model(save_path)
    print(f"Model saved to: {save_path}")
    

