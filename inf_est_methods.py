import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils import *
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




def gradient_influence_estimation(
    tr_grad_dict,
    val_grad_dict,
    hvp_cal="DataInf",
    hyperparams=None,
    device="cuda",
):
    if hyperparams is None:
        hyperparams = {}

    lambda_const_param = float(hyperparams.get("lambda_const_param", 10))
    n_iteration = int(hyperparams.get("n_iteration", 10))
    alpha_const = float(hyperparams.get("alpha_const", 1.0))

    print(f"Calculating influence with {hvp_cal}.")
    print(
        f"Params: lambda_const_param={lambda_const_param}, "
        f"n_iteration={n_iteration}, "
        f"alpha_const={alpha_const}"
    )

    train_ids = list(tr_grad_dict.keys())
    val_ids = list(val_grad_dict.keys())
    n_train = len(train_ids)

    first_tr = next(iter(tr_grad_dict.values()))
    first_val = next(iter(val_grad_dict.values()))

    param_order = [
        name for name in first_tr.keys()
        if name in first_val
    ]

    def flatten_grad_dict(example_grad_dict):
        return torch.cat([
            example_grad_dict[name].reshape(-1)
            for name in param_order
        ])

    G_train = torch.stack([
        flatten_grad_dict(tr_grad_dict[tr_id])
        for tr_id in train_ids
    ]).to(device)

    G_val = torch.stack([
        flatten_grad_dict(val_grad_dict[val_id])
        for val_id in val_ids
    ]).to(device)

    # Fast paths where flattening is directly valid
    if hvp_cal == "GradDot":
        scores = -(G_val @ G_train.T)

    elif hvp_cal == "TracIn":
        if "eta" not in hyperparams:
            raise ValueError("TracIn requires hyperparameter 'eta'.")

        eta = float(hyperparams["eta"])
        scores = -eta * (G_val @ G_train.T)

    elif hvp_cal == "GradCos":
        dots = G_val @ G_train.T
        val_norms = torch.linalg.norm(G_val, dim=1, keepdim=True)
        tr_norms = torch.linalg.norm(G_train, dim=1, keepdim=True).T
        scores = -(dots / (val_norms * tr_norms + 1e-12))

    elif hvp_cal == "DataInf":
        # DataInf needs lambda per parameter block to match your original logic.
        hvp_parts = []

        for weight_name in tqdm(param_order):
            Gt = torch.stack([
                tr_grad_dict[tr_id][weight_name].reshape(-1)
                for tr_id in train_ids
            ]).to(device)

            Gv = torch.stack([
                val_grad_dict[val_id][weight_name].reshape(-1)
                for val_id in val_ids
            ]).to(device)

            lambda_const = Gt.pow(2).mean(dim=1).mean() / lambda_const_param

            denom = lambda_const + Gt.pow(2).sum(dim=1)          # [N_train]
            C = (Gv @ Gt.T) / denom.unsqueeze(0)                 # [N_val, N_train]

            hvp = Gv / lambda_const - (C @ Gt) / (n_train * lambda_const)

            hvp_parts.append(hvp)

            del Gt, Gv, C, hvp

        H_val = torch.cat(hvp_parts, dim=1)
        scores = -(H_val @ G_train.T)

        del H_val, hvp_parts

    elif hvp_cal == "LiSSA":
        hvp_parts = []

        for weight_name in tqdm(param_order):
            Gt = torch.stack([
                tr_grad_dict[tr_id][weight_name].reshape(-1)
                for tr_id in train_ids
            ]).to(device)

            Gv = torch.stack([
                val_grad_dict[val_id][weight_name].reshape(-1)
                for val_id in val_ids
            ]).to(device)

            lambda_const = Gt.pow(2).mean(dim=1).mean() / lambda_const_param

            running_hvp = Gv.clone()

            for _ in range(n_iteration):
                dots = running_hvp @ Gt.T                         # [N_val, N_train]
                hvp_tmp = (dots @ Gt + lambda_const * running_hvp) / n_train / 1e3
                running_hvp = Gv + running_hvp - alpha_const * hvp_tmp

            hvp_parts.append(running_hvp)

            del Gt, Gv, running_hvp

        H_val = torch.cat(hvp_parts, dim=1)
        scores = -(H_val @ G_train.T)

        del H_val, hvp_parts

    
    elif hvp_cal == "Original":
        hvp_parts = []

        for weight_name in tqdm(param_order):
            Gt = torch.stack([
                tr_grad_dict[tr_id][weight_name].reshape(-1)
                for tr_id in train_ids
            ]).to(device)

            Gv = torch.stack([
                val_grad_dict[val_id][weight_name].reshape(-1)
                for val_id in val_ids
            ]).to(device)

            lambda_const = Gt.pow(2).mean(dim=1).mean() / lambda_const_param

            AAt_matrix = Gt.T @ Gt

            # AAt is symmetric, so eigh is better than eig here.
            L, V = torch.linalg.eigh(AAt_matrix)

            hvp = Gv @ V
            hvp = (hvp / (lambda_const + L / n_train)) @ V.T

            hvp_parts.append(hvp)

            del Gt, Gv, AAt_matrix, L, V, hvp

        H_val = torch.cat(hvp_parts, dim=1)
        scores = -(H_val @ G_train.T)

        del H_val, hvp_parts
        
    else:
        raise Exception("Invalid hvp calculation option.")

    df = pd.DataFrame(
        scores.detach().cpu().numpy(),
        index=val_ids,
        columns=train_ids,
        dtype=float,
    )

    del G_train, G_val, scores

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("End of influence estimation.")
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


