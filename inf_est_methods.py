import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils import *
import json




def similarity_influence_estimation(test_vec, train_vecs, hvp_cal = "RepEucSim"):
    RepCosSim = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    RepDotSim = lambda a, b: np.dot(a, b)
    RepEucSim = lambda a, b: -np.linalg.norm(a - b)**2

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
        
    elif hvp_cal == "TracInAdam":
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

            # v_hat = v / (1.0 - beta2 ** step)
            v_hat = torch.ones_like(v)
            eps = 0.0
            # m_hat = m # no bias correction / (1.0 - beta1 ** step)
            adam_update = lr / (torch.sqrt(v_hat) + eps)

            Gt = torch.stack([
                tr_grad_dict[tr_id][weight_name].reshape(-1)
                for tr_id in train_ids
            ]).to(device)

            adam_preconditioned_train = Gt * adam_update[None, :]

            train_update_parts.append(adam_preconditioned_train)

        G_train_adam = torch.cat(train_update_parts, dim=1)

        scores = -(G_val @ G_train_adam.T)

    elif hvp_cal == "TracIn":
        if "eta" not in hyperparams:
            raise ValueError("TracIn requires hyperparameter 'eta'.")

        eta = float(hyperparams["eta"])
        scores = -eta * (G_val @ G_train.T)

    elif hvp_cal == "GradCos":
        val_norms = torch.linalg.norm(G_val, dim=1, keepdim=True)
        tr_norms = torch.linalg.norm(G_train, dim=1, keepdim=True).T
        scores = -(G_val @ G_train.T / (val_norms * tr_norms + 1e-12))

    elif hvp_cal == "DataInf":

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

    elif hvp_cal == "thetaRelatIF":

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


        scores = -(H_val @ G_train.T)

        train_norms = H_train.norm(dim=1)   # ||H^{-1} g_i||, norm of val grad is ommited since we care only about ranking

        scores = scores / (train_norms.unsqueeze(0) + 1e-12)

        del H_val, hvp_parts

    elif hvp_cal == "lRelatIF":

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


        scores = -(H_val @ G_train.T)

        self_influence = torch.abs(torch.diag(H_train @ G_train.T))

        scores = scores / (
            torch.sqrt(self_influence).unsqueeze(0) + 1e-12
        )

        print(f"scores shape: {scores.shape}")
        print(f"self_influence shape: {self_influence.shape}")

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

