from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from utils import *
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import argparse
import warnings
import os
import torch
import random
import numpy as np
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig
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

    target_modules = ['q_proj', 'v_proj']
    
    MODELS = {
        "Llama": "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen4": "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen1.5": "Qwen/Qwen2-1.5B-Instruct",
        "Olmo": "allenai/OLMo-2-0425-1B-SFT",
        "randomOlmo": "allenai/OLMo-2-0425-1B-SFT",
        "Olmo7B": "allenai/OLMo-2-1124-7B-Instruct",
    }

    
    if args.model in MODELS.keys():
        model_name = MODELS[args.model]
    else:
        raise ValueError("Invalid model name")

    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise ValueError(f"{model_name} does not have a chat_template.")
    

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




    peft_config = LoraConfig(task_type="CAUSAL_LM",
                            inference_mode=False, 
                            target_modules=target_modules,
                            r=args.lora_r,
                            lora_alpha=args.lora_alpha,                                 
                            lora_dropout=0.05)
    

    if args.model == "randomOlmo":
        model = get_peft_model(model, peft_config)
        model.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")
        sys.exit()
    

    training_args = SFTConfig(
        output_dir=save_path,
        per_device_train_batch_size=args.batch_size,
        # gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=2e-4,
        logging_steps=args.logging_step,
        num_train_epochs=args.epochs,
        save_total_limit=10, 
        save_steps=10,
        max_length=args.max_length,
        report_to="none",
        completion_only_loss=True,
        optim="adamw_torch",
        seed=seed,
        data_seed=seed,
        
    )

    
    dataset = load_from_disk("datasets/" + args.dataset)


    train_dataset = dataset["train"].map(
        format_for_sft,
        remove_columns=dataset["train"].column_names,
    )

    

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config
    )

    trainer.train()
    
    
    print("Training completed.")
    trainer.save_model(save_path)
    print(f"Model saved to: {save_path}")
