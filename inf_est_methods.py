import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils import *
import json
from rank_bm25 import BM25Okapi



def random_influence_estimation(dataset, metrics_path):

    print(f"Calculating random influence...")

    train_var = dataset["train"]["variation"]
    eval_var = dataset["test"]["variation"]
    N = len(train_var)

    # counts in train set
    var_counts = {}
    for v in train_var:
        var_counts[v] = var_counts.get(v, 0) + 1

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

    print("Variation Acc:", metrics["overall"]["variation"]["accuracy"])
    print("Variation Cover:", metrics["overall"]["variation"]["coverage"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)



def gradient_influence_estimation(
    tr_grad_dict,
    val_grad_dict,
    inf_method="DataInf",
    hyperparams=None,
    device="cuda",
):
    if hyperparams is None:
        hyperparams = {}

    lambda_const_param = float(hyperparams.get("lambda_const_param", 10))
    n_iteration = int(hyperparams.get("n_iteration", 10))
    alpha_const = float(hyperparams.get("alpha_const", 1.0))

    print(f"Calculating influence with {inf_method}.")
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

    G_train = torch.stack([
        torch.cat([
            tr_grad_dict[tr_id][name].reshape(-1)
            for name in param_order
        ])
        for tr_id in train_ids
    ]).to(device)

    G_val = torch.stack([
        torch.cat([
            val_grad_dict[val_id][name].reshape(-1)
            for name in param_order
        ])
        for val_id in val_ids
    ]).to(device)


    if inf_method == "GradDot":
        scores = G_val @ G_train.T
        
    elif inf_method == "TracInAdam":
        adamw_state = hyperparams.get("adamw_optimizer_state")
        if adamw_state is None:
            raise ValueError("TracInAdam requires 'adamw_optimizer_state'.")

        train_update_parts = []

        for weight_name in param_order:
            st = adamw_state[weight_name]

            v = st["exp_avg_sq"].to(device).reshape(-1)
            m = st["exp_avg"].to(device).reshape(-1)

            step = st["step"]
            step = step.item() if torch.is_tensor(step) else int(step)

            lr = float(st["lr"])
            beta1, beta2 = st["betas"]
            eps = float(st["eps"])
            wd = float(st["weight_decay"])

            if wd != 0.0:
                raise ValueError(
                    f"TracInAdam currently assumes weight_decay == 0. "
                    f"Found weight_decay={wd} for parameter '{weight_name}'."
                )
            

            Gt = torch.stack([
                tr_grad_dict[tr_id][weight_name].reshape(-1)
                for tr_id in train_ids
            ]).to(device)

            updated_m = beta1 * m + (1 - beta1) * Gt
            updated_v = beta2 * v + (1 - beta2) * Gt ** 2

            updated_m_hat = updated_m / (1 - beta1 ** (step + 1))
            updated_v_hat = updated_v / (1 - beta2 ** (step + 1))

            adam_preconditioned_train = lr * updated_m_hat / (torch.sqrt(updated_v_hat) + eps)

            train_update_parts.append(adam_preconditioned_train)

        G_train_adam = torch.cat(train_update_parts, dim=1)

        scores = G_val @ G_train_adam.T

    elif inf_method == "TracIn":

        adamw_state = hyperparams.get("adamw_optimizer_state")
        if adamw_state is None:
            raise ValueError("TracIn requires 'adamw_optimizer_state'.")
        
        lrs = {st["lr"] for st in adamw_state.values()}

        if len(lrs) != 1:
            raise ValueError(
                f"Expected same learning rate for all weights, got: {sorted(lrs)}"
            )
        
        lr = list(lrs)[0]

        scores = lr * (G_val @ G_train.T)

    elif inf_method == "GradCos":
        val_norms = torch.linalg.norm(G_val, dim=1, keepdim=True)
        tr_norms = torch.linalg.norm(G_train, dim=1, keepdim=True).T
        scores = G_val @ G_train.T / (val_norms * tr_norms + 1e-12)

    elif inf_method == "DataInf":

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
        scores = H_val @ G_train.T

        del H_val, hvp_parts

    elif inf_method == "theta-RelatIF":

        hvp_parts = []
        train_hvp_parts = []

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
            C_val = (Gv @ Gt.T) / denom.unsqueeze(0)
            hvp_val = Gv / lambda_const - (C_val @ Gt) / (n_train * lambda_const)

            C_train = (Gt @ Gt.T) / denom.unsqueeze(0)
            hvp_train = Gt / lambda_const - (C_train @ Gt) / (n_train * lambda_const)

            hvp_parts.append(hvp_val)
            train_hvp_parts.append(hvp_train)

        H_val = torch.cat(hvp_parts, dim=1)
        H_train = torch.cat(train_hvp_parts, dim=1)


        scores = H_val @ G_train.T

        train_norms = H_train.norm(dim=1)   # ||H^{-1} g_i||, norm of val grad is ommited since we care only about ranking

        scores = scores / (train_norms.unsqueeze(0) + 1e-12)

        del H_val, hvp_parts

    elif inf_method == "l-RelatIF":

        hvp_parts = []
        train_hvp_parts = []

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
            C_val = (Gv @ Gt.T) / denom.unsqueeze(0)
            hvp_val = Gv / lambda_const - (C_val @ Gt) / (n_train * lambda_const)

            C_train = (Gt @ Gt.T) / denom.unsqueeze(0)
            hvp_train = Gt / lambda_const - (C_train @ Gt) / (n_train * lambda_const)

            hvp_parts.append(hvp_val)
            train_hvp_parts.append(hvp_train)

        H_val = torch.cat(hvp_parts, dim=1)
        H_train = torch.cat(train_hvp_parts, dim=1)


        scores = H_val @ G_train.T

        self_influence = torch.abs(torch.diag(H_train @ G_train.T))

        scores = scores / (
            torch.sqrt(self_influence).unsqueeze(0) + 1e-12
        )

        print(f"scores shape: {scores.shape}")
        print(f"self_influence shape: {self_influence.shape}")

        del H_val, hvp_parts

    elif inf_method == "LiSSA":
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
        scores = H_val @ G_train.T

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



def BM25_scores(dataset):

    print("Calculating BM25 scores...")

    train_texts = [
        p + " " + r
        for p, r in zip(
            dataset["train"]["prompts"],
            dataset["train"]["response"]
        )
    ]

    test_texts = [
        p + " " + r
        for p, r in zip(
            dataset["test"]["prompts"],
            dataset["test"]["response"]
        )
    ]

    tokenized_train = [x.lower().split() for x in train_texts]
    tokenized_test = [x.lower().split() for x in test_texts]

    bm25 = BM25Okapi(tokenized_train)

    scores = []
    for q in tqdm(tokenized_test):
        scores.append(bm25.get_scores(q))

    bm25_df = pd.DataFrame(scores)

    return bm25_df



def RepSim(
    model,
    tokenizer,
    chat_template,
    train_prompts,
    test_prompts,
    device="cuda"
):


    chat_template = chat_template.replace("{response}", "")

    def get_hidden_states(prompts):
        reps = []

        for p in tqdm(prompts):
            inputs = tokenizer(
                chat_template.format(prompt=p),
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            reps.append(
                outputs.hidden_states[-1][:, -1, :]
                .view(-1)
                .float()
                .cpu()
                .numpy()
            )

        reps = np.asarray(reps)
        norms = np.linalg.norm(reps, axis=1, keepdims=True)
        reps = reps / np.clip(norms, 1e-12, None)
        return reps

    model.eval()

    print("Generating test representations...")
    test_reps = get_hidden_states(test_prompts)

    print("Generating train representations...")
    train_reps = get_hidden_states(train_prompts)

    sim_matrix = test_reps @ train_reps.T

    return pd.DataFrame(sim_matrix)



def ekfac_influence_estimation(tokenizer,
                               model,
                            dataset,
                            max_length = 128,
                            batch_size = 10,
                            output_dir="results/EKFAC",
                            factor_strategy = "ekfac"):


    autoregressive = True
    device = "cuda"

    model.to(device)
    model.eval()  

    task = KronfluenceTask(model, autoregressive = autoregressive)
    model = prepare_model(model=model, task=task)


    analyzer = Analyzer(analysis_name="ekfac_analysis_backdoor", model=model, task=task,
                        output_dir=output_dir)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")  
    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factor_args = FactorArguments(strategy=factor_strategy)

    chat_template = f"[INST] {{prompt}} [/INST] {{response}}"

    tokenized_tr = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=max_length)    
    tokenized_val = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=max_length)


    analyzer.fit_all_factors(factors_name=factor_strategy, 
                         dataset=tokenized_tr,
                         factor_args=factor_args,
                         overwrite_output_dir=True,
                         per_device_batch_size=batch_size,
                         initial_per_device_batch_size_attempt=batch_size,)


    # Configure parameters for DataLoader.
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)



    analyzer.compute_pairwise_scores(
    score_args = score_args,
    scores_name=factor_strategy,
    factors_name=factor_strategy,
    query_dataset=tokenized_val,
    train_dataset=tokenized_tr,
    per_device_query_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    overwrite_output_dir=True,
)


    scores = analyzer.load_pairwise_scores(scores_name=factor_strategy)["all_modules"]
    print(f"Scores shape: {scores.shape}")
    print(f"Saved to: {factor_strategy}")

    return scores


