from datasets import load_from_disk
from transformers import AutoTokenizer
import time


from utils import *
from postprocess_utils import *
from inf_est_methods import *
from tqdm.auto import tqdm


import pickle
import argparse
import os
import sys
import glob
from huggingface_hub import login

with open("TOKENS.txt", "r") as f:
    line = f.read().strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--hvp_cal', type=str, required=False, help='influence estimation method')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--lambda_c', type=float, default=10, help='lambda const')
    parser.add_argument('--iter', type=int, default=3, help='#iteration')
    parser.add_argument('--alpha', type=float, default=1., help='alpha_const')
    parser.add_argument('--inf_args', type=str, required=False, help='Other args, method-specific.')
    args = parser.parse_args()

    if args.model in {"Llama", "Qwen0.5", "Qwen1.5", "Olmo"}:
        model_name, chat_template = template_setting(args.model)
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
    metrics_filename = f"{args.dataset}_{args.model}_{args.hvp_cal}_metrics_results.json".replace(" ", "_")
    metrics_path = os.path.join(results_dir, metrics_filename)

    quantization_config = None
    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, 
                                                      device_map='auto', attn_implementation="eager")

    start_time = time.time()
    
    if args.hvp_cal == "ekfac":
        # these are from sbatch file
        HVP_METHOD="ekfac"
        INF_ARGS="lambda_const_param=0.01,n_iteration=10,alpha_const=1."

        
        model = PeftModel.from_pretrained(
            base_model,
            "lora_adapter/" + core_path
        )

        model = model.merge_and_unload()

        # full fine-tuned model:
        # 
        # model_path = f"finetuned_model/{args.model}/{args.dataset}_{args.epochs}"

        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     dtype=torch.float32,
        # )
        # model.config.use_cache = False    

        
        config = {
            "model": {
                "family": "tinyllama",
                "num_layers": model.config.num_hidden_layers,
            }
        }

        scores = ekfac_influence_estimation(tokenizer,
                                            model = model,
                                            dataset = dataset,
                                            config = config,
                                            max_length = 128,
                                            output_dir="results/EKFAC",
                                            use_half_precision = False,
                                            use_compile = False,
                                            query_gradient_rank = -1,
                                            save_id = True,
                                            factor_strategy = "diagonal")

        influence_inf = pd.DataFrame(-scores.cpu().numpy())

    elif (args.hvp_cal == "random"):

        print(f"Calculating random influence...")
        
        random_influence_estimation(dataset = dataset, metrics_path = metrics_path)

        sys.exit()



    elif "rep" in args.hvp_cal and "sim" in args.hvp_cal:

        
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

            check.append(outputs['hidden_states'][-1][:, -1, :].view(-1).cpu().numpy().T)

        query = []
        for p in tqdm(dataset['train']['prompts']):
            inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                query.append(outputs['hidden_states'][-1][:, -1, :].view(-1).cpu().numpy())


        sim_matrix = []
        for item in tqdm(check):
            arr = similarity_influence_estimation(test_vec=item, train_vecs=query,hvp_cal=args.hvp_cal)
            sim_matrix.append(arr)

        sim_df = pd.DataFrame(sim_matrix)

        influence_inf = -sim_df  #negation because the larger the similarity, the better; unlike influence.

    

    elif args.hvp_cal == "TracIn":

        print("Calculating TracIn...")

        ckpt_root = "lora_adapter/" + core_path

        checkpoint_paths = sorted(
            glob.glob(os.path.join(ckpt_root, "checkpoint-*")),
            key=lambda x: int(x.split("-")[-1])
        ) # [:1] for testing just the 1 check point



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

            
            eta = get_eta_from_trainer_state(ckpt_path)

            checkpoint_influence = gradient_influence_estimation(
                tr_grad_dict=tr_grad_dict,
                val_grad_dict=val_grad_dict,
                hvp_cal="TracIn",
                hyperparams={"eta": eta},
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
                                                      hvp_cal=args.hvp_cal, 
                                                      hyperparams = inf_args_map)

    end_time = time.time()

    cache_dir = 'cache/' + args.model + '/'
    os.makedirs(cache_dir, exist_ok=True)
    influence_inf.to_csv(cache_dir + args.dataset + '_' + str(args.epochs) + args.hvp_cal + '.csv', index_label=False)
    run_benchmark_measures(influence = influence_inf, train_dataset = dataset['train'], 
                  validation_dataset = dataset['test'], 
                  metrics_path = metrics_path,
                  time_elapsed = end_time - start_time)


