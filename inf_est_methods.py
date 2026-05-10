import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import json




def similarity_influence_estimation(test_vec, train_vecs, hvp_cal = "rep_cos_sim"):
    rep_cos_sim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    rep_dot_sim = lambda a, b: np.dot(a, b)
    rep_euc_sim = lambda a, b: -np.linalg.norm(a - b)**2

    sim_fn = locals().get(hvp_cal)
    if sim_fn is None:
        raise ValueError(f"Unknown similarity type: {hvp_cal}")
    
    sim = []
    for i in range(len(train_vecs)):
        sim.append(sim_fn(test_vec, train_vecs[i]))
    return np.array(sim)




def gradient_influence_estimation(tr_grad_dict, val_grad_dict, hvp_cal='DataInf', hyperparams = None):
    
    if hyperparams is None:
        hyperparams = {}
    
  
    lambda_const_param = float(hyperparams.get("lambda_const_param", 10)) #default
    n_iteration = int(hyperparams.get("n_iteration", 10)) #default
    alpha_const = float(hyperparams.get("alpha_const", 1.0)) #default


    print(f"Calculating influence with {hvp_cal}.")
    print(
        f"Params: lambda_const_param={lambda_const_param}, "
        f"n_iteration={n_iteration}, "
        f"alpha_const={alpha_const}"
    )

    hvp_dict = defaultdict(dict)
    IF_dict = defaultdict(dict)
    n_train = len(tr_grad_dict.keys())

    def calculate_lambda_const(tr_grad_dict, weight_name):
        S = torch.zeros(len(tr_grad_dict.keys()))
        for tr_id in tr_grad_dict:
            tmp_grad = tr_grad_dict[tr_id][weight_name]
            S[tr_id] = torch.mean(tmp_grad**2)

        return torch.mean(S) / lambda_const_param

    if hvp_cal == 'Original':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)

                AAt_matrix = torch.zeros(torch.outer(tr_grad_dict[0][weight_name].reshape(-1), 
                                                     tr_grad_dict[0][weight_name].reshape(-1)).shape)
                for tr_id in tr_grad_dict:
                    tmp_mat = torch.outer(tr_grad_dict[tr_id][weight_name].reshape(-1), 
                                          tr_grad_dict[tr_id][weight_name].reshape(-1))
                    AAt_matrix += tmp_mat

                L, V = torch.linalg.eig(AAt_matrix)
                L, V = L.float(), V.float()
                hvp = val_grad_dict[val_id][weight_name].reshape(-1) @ V
                hvp = (hvp / (lambda_const + L / n_train)) @ V.T
                hvp_dict[val_id][weight_name] = hvp.reshape(len(tr_grad_dict[0][weight_name]), -1)
                del tmp_mat, AAt_matrix, V

    elif hvp_cal == 'DataInf':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)

                hvp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                for tr_id in tr_grad_dict:
                    tmp_grad = tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (val_grad_dict[val_id][weight_name] - C_tmp * tmp_grad) / (n_train * lambda_const)
                
                hvp_dict[val_id][weight_name] = hvp

    elif hvp_cal == 'LiSSA':
        for val_id in tqdm(val_grad_dict.keys()):
            for weight_name in val_grad_dict[val_id]:
                lambda_const = calculate_lambda_const(tr_grad_dict, weight_name)

                running_hvp = val_grad_dict[val_id][weight_name]
                for _ in range(n_iteration):
                    hvp_tmp = torch.zeros(val_grad_dict[val_id][weight_name].shape)
                    for tr_id in tr_grad_dict:
                        tmp_grad = tr_grad_dict[tr_id][weight_name]
                        hvp_tmp += (torch.sum(tmp_grad * running_hvp) * tmp_grad + lambda_const * running_hvp) / n_train / 1e3
                    
                    running_hvp = val_grad_dict[val_id][weight_name] + running_hvp - alpha_const * hvp_tmp

                hvp_dict[val_id][weight_name] = running_hvp

    
    elif hvp_cal == 'GradDot' or hvp_cal == "GradCos":
        hvp_dict = val_grad_dict.copy()
    else:
        raise Exception("Invalid hvp calculation option.")

    for tr_id in tr_grad_dict:
        for val_id in val_grad_dict:
            if_tmp_value = 0
            norm_tr = 0
            norm_val_hvp = 0
            for weight_name in val_grad_dict[0]:

                g_val_hvp = hvp_dict[val_id][weight_name]
                g_tr = tr_grad_dict[tr_id][weight_name]
                if_tmp_value += torch.sum(g_val_hvp * g_tr)

                # for normalization
                norm_val_hvp += torch.sum(g_val_hvp * g_val_hvp)
                norm_tr += torch.sum(g_tr * g_tr)

            if hvp_cal == "GradCos":
                cos_sim = if_tmp_value / (torch.sqrt(norm_tr) * torch.sqrt(norm_val_hvp) + 1e-12)
                IF_dict[tr_id][val_id] = -cos_sim
            else:
                IF_dict[tr_id][val_id] = -if_tmp_value

    print("End of influence estimation.")
    return pd.DataFrame(IF_dict, dtype=float)




def random_influence_estimation(dataset, metrics_path):
    
    train_var = dataset["train"]["variation"]
    eval_var = dataset["test"]["variation"]
    N = len(train_var)

    has_subvariation = (
        "subvariation" in dataset["train"].column_names
        and "subvariation" in dataset["test"].column_names
    )

    # counts in train set
    var_counts = {}
    for v in train_var:
        var_counts[v] = var_counts.get(v, 0) + 1

    if has_subvariation:
        train_subvar = dataset["train"]["subvariation"]
        eval_subvar = dataset["test"]["subvariation"]

        subvar_counts = {}
        for s in train_subvar:
            subvar_counts[s] = subvar_counts.get(s, 0) + 1

    # overall expected metrics
    var_values = [var_counts[v] / N for v in eval_var]

    metrics = {
        "overall": {
            "variation": {
                "accuracy": float(np.mean(var_values)),
                "coverage": float(np.mean(var_values)),
            }
        },
        "per_variation": {},
    }

    # per-variation expected metrics
    eval_var_counts = {}
    for v in eval_var:
        eval_var_counts[v] = eval_var_counts.get(v, 0) + 1

    for v, count in eval_var_counts.items():
        p = var_counts.get(v, 0) / N
        metrics["per_variation"][str(v)] = {
            "num_samples": count,
            "accuracy": float(p),
            "coverage": float(p),
        }

    if has_subvariation:
        subvar_values = [subvar_counts[s] / N for s in eval_subvar]
        subvar_acc_rate = np.mean(subvar_values)
        subvar_cov_rate = np.mean(subvar_values)

        metrics["overall"]["subvariation"] = {
            "accuracy": float(subvar_acc_rate),
            "coverage": float(subvar_cov_rate),
        }

        metrics["per_subvariation"] = {}

        eval_subvar_counts = {}
        for s in eval_subvar:
            eval_subvar_counts[s] = eval_subvar_counts.get(s, 0) + 1

        for s, count in eval_subvar_counts.items():
            p = subvar_counts.get(s, 0) / N
            metrics["per_subvariation"][str(s)] = {
                "num_samples": count,
                "accuracy": float(p),
                "coverage": float(p),
            }

    print("Variation Acc:", metrics["overall"]["variation"]["accuracy"])
    print("Variation Cover:", metrics["overall"]["variation"]["coverage"])

    if has_subvariation:
        print("Subvariation Acc:", metrics["overall"]["subvariation"]["accuracy"])
        print("Subvariation Cover:", metrics["overall"]["subvariation"]["coverage"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


# TRACIN
from collections import defaultdict
import torch
import pandas as pd


from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd


def TracIn_Adam(
    checkpoint_gradients,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    device="cuda",
):

    train_ids = None
    val_ids = None

    def align_shape(x, target):
        if x.shape == target.shape:
            return x
        if x.T.shape == target.shape:
            return x.T
        raise ValueError(f"Shape mismatch: {x.shape} vs {target.shape}")

    def flatten_grad_dict(example_grad_dict, param_order):
        parts = []
        for name in param_order:
            g = example_grad_dict[name]
            parts.append(g.reshape(-1))
        return torch.cat(parts)

    eta, tr_grad_dict, val_grad_dict, adam_optimizer_state = checkpoint_gradients

    train_ids = list(tr_grad_dict.keys())
    val_ids = list(val_grad_dict.keys())

    # Use only params present in both train and val gradients
    param_order = [
        name for name in next(iter(tr_grad_dict.values())).keys()
        if name in next(iter(val_grad_dict.values()))
    ]

    # Build raw gradient matrices
    G_train = torch.stack([
        flatten_grad_dict(tr_grad_dict[tr_id], param_order)
        for tr_id in train_ids
    ]).to(device)

    G_val = torch.stack([
        flatten_grad_dict(val_grad_dict[val_id], param_order)
        for val_id in val_ids
    ]).to(device)

    # Flatten Adam state in the same parameter order
    m_parts = []
    v_parts = []

    for name in param_order:
        ref = next(iter(tr_grad_dict.values()))[name]

        m = adam_optimizer_state[name]["exp_avg"]
        v = adam_optimizer_state[name]["exp_avg_sq"]

        m = align_shape(m, ref)
        v = align_shape(v, ref)

        m_parts.append(m.reshape(-1))
        v_parts.append(v.reshape(-1))

    m = torch.cat(m_parts).to(device)
    v = torch.cat(v_parts).to(device)

    # Adam-precondition both train and validation gradients
    G_train = (beta1 * m + (1 - beta1) * G_train) / torch.sqrt(
        beta2 * v + (1 - beta2) * G_train.pow(2) + eps
    )

    G_val = (beta1 * m + (1 - beta1) * G_val) / torch.sqrt(
        beta2 * v + (1 - beta2) * G_val.pow(2) + eps
    )

    # Cosine similarity, matching normalized LESS-style scoring
    G_train = F.normalize(G_train, dim=1)
    G_val = F.normalize(G_val, dim=1)

    # Shape: [N_train, N_val]
    scores = -eta * (G_train @ G_val.T)


    del G_train, G_val
    torch.cuda.empty_cache()

    df = pd.DataFrame(
        scores.T.detach().cpu().numpy(),
        index=val_ids,
        columns=train_ids,
        dtype=float,
    )

    print(df.shape)

    return df


"""
EK-FAC
"""

from datetime import timedelta
from transformers import default_data_collator
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM

import kronfluence
from kronfluence.analyzer import Analyzer
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs


from kronfluence.arguments import ScoreArguments, FactorArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments


from utils import LanguageModelingTask, get_preprocessed_dataset


def ekfac_influence_estimation(tokenizer,
                               model,
                            dataset,
                            config,
                            max_length = 128,
                            output_dir="results/EKFAC",
                            use_half_precision = False,
                            use_compile = False,
                            query_gradient_rank = -1,
                            save_id = True,
                            factor_strategy = "diagonal",
                            chat_template = f"[INST] {{prompt}} [/INST] {{response}}"):


    
    

    task = LanguageModelingTask(config)
    model = kronfluence.prepare_model(model, task)


    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    if use_compile:
        model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name=factor_strategy,
        model=model,
        task=task,
        profile=False,
        output_dir=output_dir
    )


    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)



    tokenized_tr = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=max_length)    
    tokenized_val = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=max_length)



    factor_args = FactorArguments(strategy=factor_strategy)
    if use_half_precision:
        factor_args = all_low_precision_factor_arguments(
            strategy=factor_strategy,
            dtype=torch.float16
        )
        factor_strategy += "_half"
    if use_compile:
        factor_strategy += "_compile"

    factor_args.covariance_max_examples = 1
    factor_args.lambda_max_examples = 1

    analyzer.fit_all_factors(
        factors_name=factor_strategy,
        dataset=tokenized_tr,
        per_device_batch_size=1,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )



    ##### Compute pairwise scores. #####
    score_args = ScoreArguments()
    scores_name = "default_scores"


    if use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"

    if use_compile:
        scores_name += "_compile"

    rank = query_gradient_rank if query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"

    if save_id:
        scores_name += f"_{save_id}"


    score_args.compute_per_token_scores = False
    score_args.aggregate_query_gradients = False

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factor_strategy,   
        query_dataset=tokenized_val,
        train_dataset=tokenized_tr,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        overwrite_output_dir=True,
    )



    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    print(f"Scores shape: {scores.shape}")
    print(f"Saved to: {scores_name}")

    return scores




