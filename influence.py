# from importlib.metadata import distributions

# for dist in distributions():
#     print(f"{dist.metadata['Name']}=={dist.version}")

from datasets import load_from_disk
from transformers import AutoTokenizer
import time

from utils import *
from postprocess_utils import *
from inf_est_methods import *
from tqdm.auto import tqdm

import random
import pickle
import argparse
import os
import sys
import glob
from huggingface_hub import login

seed = 1

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

with open("settings_txt/TOKENS.txt", "r") as f:
    line = f.read().strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--inf_method', type=str, required=False, help='influence estimation method')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--lambda_c', type=float, default=10, help='lambda const')
    parser.add_argument('--iter', type=int, default=3, help='#iteration')
    parser.add_argument('--alpha', type=float, default=1., help='alpha_const')
    parser.add_argument('--inf_args', type=str, required=False, help='Other args, method-specific.')
    args = parser.parse_args()


    with open("settings_txt/target_modules.txt") as f:
        target_modules = [
            line.strip()
            for line in f
            if line.strip()
        ]

    
    chat_template, model_name = template_setting(args.model)

    core_path = f"{args.model}/{args.dataset}_{args.epochs}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    dataset = load_from_disk("datasets/" + args.dataset)

    # results statistics directory
    results_dir = "results/json"
    os.makedirs(results_dir, exist_ok=True)
    metrics_filename = f"{args.dataset}_{args.model}_{args.inf_method}_metrics_results.json".replace(" ", "_")
    metrics_path = os.path.join(results_dir, metrics_filename)

    start_time = time.time()

    if args.inf_method == "random":
        random_influence_estimation(dataset = dataset, metrics_path = metrics_path)

        sys.exit()

    elif args.inf_method == "BM25":

        influence_inf = BM25_scores(dataset = dataset)

    elif args.inf_method == "RepSim":

        base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      device_map='cuda')

        model = PeftModel.from_pretrained(
                base_model,
                "lora_adapter/" + core_path
            )
        
        model.config.use_cache = False
        model.to("cuda")
        model.eval()

        influence_inf = RepSim(
        model=model,
        tokenizer=tokenizer,
        chat_template=chat_template,
        train_prompts=dataset["train"]["prompts"],
        test_prompts=dataset["test"]["prompts"],
    )

    else:

        base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      device_map='cuda')


        tokenized_tr = get_preprocessed_dataset(
            tokenizer, dataset["train"], chat_template, max_length=args.max_length
        )
        tokenized_val = get_preprocessed_dataset(
            tokenizer, dataset["test"], chat_template, max_length=args.max_length
        )

                
        if args.inf_method == "EKFAC":
                
                model = PeftModel.from_pretrained(
                    base_model,
                    "lora_adapter/" + core_path,
                    is_trainable=True
                )

                model.config.use_cache = False
                model.to("cuda")
                
                
                scores = ekfac_influence_estimation(tokenizer,
                                                    model,
                                                    tokenized_tr,
                                                    tokenized_val,
                                                    output_dir="results/EKFAC",
                                                    factor_strategy = "ekfac",
                                                    target_modules = target_modules)

                influence_inf = pd.DataFrame(scores.detach().float().cpu().numpy())

            

        elif "TracIn" in args.inf_method:

            if "random" in args.model:
                print("TracIn disabled for randomized models.")
                sys.exit()

            print(f"Calculating {args.inf_method}...")

            ckpt_root = "lora_adapter/" + core_path

            checkpoint_paths = sorted(
                glob.glob(os.path.join(ckpt_root, "checkpoint-*")),
                key=lambda x: int(x.split("-")[-1])
            ) #[:1] #for testing just 1 check point


            
            influence_inf = None

            for ckpt_path in tqdm(checkpoint_paths, desc="Checkpoints"):
                print(f"Collecting gradients for {ckpt_path}")

                
                model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)

                model.config.use_cache = False
                model.to("cuda")

                tr_grad_dict, val_grad_dict = collect_gradient( 
                    model,
                    tokenizer,
                    tokenized_tr,
                    tokenized_val
                )

                
                adamw_optimizer_state = load_adamw_optimizer_state(model, ckpt_path)


                checkpoint_influence = TracIn(
                    tr_grad_dict=tr_grad_dict,
                    val_grad_dict=val_grad_dict,
                    hyperparams={"adamw_optimizer_state": adamw_optimizer_state},
                )

                if influence_inf is None:
                    influence_inf = checkpoint_influence
                else:
                    influence_inf += checkpoint_influence

                del model
                torch.cuda.empty_cache()


        else:

            model = PeftModel.from_pretrained(
                    base_model,
                    "lora_adapter/" + core_path,
                    is_trainable=True
                )

            model.config.use_cache = False
            model.to("cuda")


            tr_grad_dict, val_grad_dict = collect_gradient(model, tokenizer, tokenized_tr, tokenized_val)

            inf_args_map = dict(
            item.split('=') for item in (args.inf_args.split(',') if args.inf_args else [])
            )

            method = globals()[args.inf_method]

            influence_inf = method(tr_grad_dict = tr_grad_dict, 
                                    val_grad_dict = val_grad_dict,
                                    hyperparams = inf_args_map)

    end_time = time.time()

    cache_dir = 'cache/' + args.model + '/'
    os.makedirs(cache_dir, exist_ok=True)
    influence_inf.to_csv(cache_dir + args.dataset + '_' + str(args.epochs) + args.inf_method + '.csv', index_label=False)
    run_benchmark_measures(influence = influence_inf, train_dataset = dataset['train'], 
                  validation_dataset = dataset['test'], 
                  metrics_path = metrics_path,
                  time_elapsed = end_time - start_time)


