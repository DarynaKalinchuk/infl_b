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

with open("TOKENS.txt", "r") as f:
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

    if args.model in {"Llama", "Qwen0.5", "Qwen1.5", "Olmo"}:
        model_name, chat_template = template_setting(args.model)
    elif args.model == "randomOlmo":
        model_name, chat_template = template_setting("Olmo")
    else:
        raise ValueError("Invalid model name")

    core_path = f"{args.model}/{args.dataset}_{args.epochs}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("datasets/" + args.dataset)

    # results statistics directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_filename = f"{args.dataset}_{args.model}_{args.inf_method}_metrics_results.json".replace(" ", "_")
    metrics_path = os.path.join(results_dir, metrics_filename)

    quantization_config = None
    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, 
                                                      device_map='auto')

    start_time = time.time()
    

    if (args.inf_method == "random"):
        
        random_influence_estimation(dataset = dataset, metrics_path = metrics_path)

        sys.exit()

    elif args.inf_method == "BM25":

        influence_inf = BM25_scores(dataset = dataset)


    elif "Sim" in args.inf_method:

        
        model = PeftModel.from_pretrained(
            base_model,
            "lora_adapter/" + core_path
        )
        model.eval()

        chat_template = chat_template.replace("{response}", "")

        print('Generate hidden states...')

        check = []
        for p in tqdm(dataset['test']['prompts']):
            inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            check.append(
                outputs["hidden_states"][-1][:, -1, :]
                .view(-1)
                .float()
                .cpu()
                .numpy()
            )

        query = []
        for p in tqdm(dataset['train']['prompts']):
            inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                query.append(
                    outputs["hidden_states"][-1][:, -1, :]
                    .view(-1)
                    .float()
                    .cpu()
                    .numpy()
                )


        sim_matrix = []
        for item in tqdm(check):
            arr = similarity_influence_estimation(test_vec=item, train_vecs=query,inf_method=args.inf_method)
            sim_matrix.append(arr)

        sim_df = pd.DataFrame(sim_matrix)

        influence_inf = -sim_df  #negation because the larger the similarity, the better; unlike influence.


    elif "TracIn" in args.inf_method:

        if "random" in args.model:
            print("TracIn disabled for randomized models.")
            sys.exit()

        print("Calculating {args.inf_method}...")

        ckpt_root = "lora_adapter/" + core_path

        checkpoint_paths = sorted(
            glob.glob(os.path.join(ckpt_root, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1])
        ) #[:1] #for testing just the 1 check point


        tokenized_tr = get_preprocessed_dataset(
            tokenizer, dataset["train"], chat_template, max_length=args.max_length
        )
        tokenized_val = get_preprocessed_dataset(
            tokenizer, dataset["test"], chat_template, max_length=args.max_length
        )

        influence_inf = None

        for ckpt_path in tqdm(checkpoint_paths, desc="Checkpoints"):
            print(f"Collecting gradients for {ckpt_path}")

            
            model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)

            tr_grad_dict, val_grad_dict = collect_gradient( 
                model,
                tokenizer,
                tokenized_tr,
                tokenized_val
            )

            
            adamw_optimizer_state = load_adamw_optimizer_state(model, ckpt_path)


            checkpoint_influence = gradient_influence_estimation(
                tr_grad_dict=tr_grad_dict,
                val_grad_dict=val_grad_dict,
                inf_method= args.inf_method,
                hyperparams={"adamw_optimizer_state": adamw_optimizer_state},
            )

            if influence_inf is None:
                influence_inf = checkpoint_influence
            else:
                influence_inf += checkpoint_influence


    else:


        tokenized_tr = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=args.max_length)
        tokenized_val = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=args.max_length)
        
        
        model = PeftModel.from_pretrained(base_model, "lora_adapter/" + core_path, is_trainable=True)
        
        tr_grad_dict, val_grad_dict = collect_gradient(model, tokenizer, tokenized_tr, tokenized_val)


        inf_args_map = dict(
        item.split('=') for item in (args.inf_args.split(',') if args.inf_args else [])
        )

        influence_inf = gradient_influence_estimation(tr_grad_dict = tr_grad_dict, 
                                                      val_grad_dict = val_grad_dict,
                                                      inf_method=args.inf_method, 
                                                      hyperparams = inf_args_map)

    end_time = time.time()

    cache_dir = 'cache/' + args.model + '/'
    os.makedirs(cache_dir, exist_ok=True)
    influence_inf.to_csv(cache_dir + args.dataset + '_' + str(args.epochs) + args.inf_method + '.csv', index_label=False)
    run_benchmark_measures(influence = influence_inf, train_dataset = dataset['train'], 
                  validation_dataset = dataset['test'], 
                  metrics_path = metrics_path,
                  time_elapsed = end_time - start_time)


