import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
import os
import json
import math
import re

from datasets import Dataset
from typing import Any, Dict, List, Optional, Union
from kronfluence import FactorArguments, ScoreArguments
from torch import nn

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task

from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs


def collect_gradient(model, dataloader, device="cuda", bring_to_cpu=True):

    model.to(device)
    model.eval()

    grad_dict_all = {}

    for step, batch in enumerate(tqdm(dataloader)):
        model.zero_grad(set_to_none=True)

        batch["labels"] = batch["input_ids"].clone()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad_dict = {}

        for k, v in model.named_parameters():
            if "lora_A" in k:
                grad = v.grad.detach()
                grad_dict[k] = grad.cpu() if bring_to_cpu else grad.clone()

            elif "lora_B" in k:
                grad = v.grad.detach().T
                grad_dict[k] = grad.cpu() if bring_to_cpu else grad.clone()

        grad_dict_all[step] = grad_dict

    return grad_dict_all



def load_adamw_optimizer_state(model, ckpt_path):

    optimizer_path = os.path.join(ckpt_path, "optimizer.pt")

    print(f"\nLoading optimizer state from: {optimizer_path}")

    optimizer_state = torch.load(optimizer_path, map_location="cpu")

    state = optimizer_state["state"]
    param_groups = optimizer_state["param_groups"]


    trainable_params = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad
    ]


    param_ids = []
    param_group_map = {}

    for group_idx, group in enumerate(param_groups):
        for pid in group["params"]:
            param_ids.append(pid)
            param_group_map[pid] = group_idx


    if len(trainable_params) != len(param_ids):
        raise ValueError(
            f"Param count mismatch: "
            f"model={len(trainable_params)}, "
            f"optimizer={len(param_ids)}"
        )

    adam_optimizer_state = {}


    for idx, ((name, p), pid) in enumerate(zip(trainable_params, param_ids)):

        if pid not in state:
            raise ValueError(f"Missing optimizer state for pid={pid}")

        opt_state = state[pid]

        required_keys = ["step", "exp_avg", "exp_avg_sq"]

        for key in required_keys:
            if key not in opt_state:
                raise ValueError(f"Missing '{key}' for {name}")

        exp_avg = opt_state["exp_avg"]
        exp_avg_sq = opt_state["exp_avg_sq"]
        step = opt_state["step"]

        if exp_avg.shape != p.shape:
            raise ValueError(
                f"exp_avg mismatch for {name}: "
                f"optimizer={tuple(exp_avg.shape)}, "
                f"model={tuple(p.shape)}"
            )

        if exp_avg_sq.shape != p.shape:
            raise ValueError(
                f"exp_avg_sq mismatch for {name}: "
                f"optimizer={tuple(exp_avg_sq.shape)}, "
                f"model={tuple(p.shape)}"
            )

        step_value = step.item() if torch.is_tensor(step) else step


        adam_optimizer_state[name] = {
            "step": step_value,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "lr": param_groups[param_group_map[pid]]["lr"],
            "betas": param_groups[param_group_map[pid]]["betas"],
            "eps": param_groups[param_group_map[pid]]["eps"],
            "weight_decay": param_groups[param_group_map[pid]]["weight_decay"],
        }

    print("\nOptimizer state successfully loaded and verified.")

    return adam_optimizer_state




class KronfluenceTask(Task):
    def __init__(self, model: torch.nn.Module, autoregressive: bool = False, device: str = "cuda",
                 target_modules= None):
        super().__init__()
        self.autoregressive = autoregressive
        self.device = device
        self.model = model
        self.target_modules = target_modules or [
            "q_proj",
            "v_proj",
            "modules_to_save.default.out_proj",
        ]

    def compute_train_loss(
        self,
        batch: Any,
        model: torch.nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        if self.autoregressive: # labels are inputs - teacher forcing
            batch["labels"] = batch["input_ids"]
        batch.to(self.device)
        outputs = model(**batch)
        loss = outputs.loss
        return loss

    def compute_measurement(
        self,
        batch: Any,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        collected_names = []
        for name, module in self.model.named_modules():
            if any(target_module in name for target_module in self.target_modules) \
                and isinstance(module, torch.nn.Linear) and "base_layer" not in name:
                collected_names.append(name)
        return collected_names

    def get_attention_mask(self, batch: Any) -> Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]:
        if "attention_mask" in batch:
            return batch["attention_mask"]
        return None  # Attention mask not used.
    


def format_for_sft(example, eos_token):
    prompt = example["prompts"].strip()
    response = example["response"].strip()

    example["text"] = f"{prompt} -> {response}{eos_token}"
    return example


def causal_tokenize(tokenizer, dataset, max_length=128):
    #tokenizing prompts only?
    tokenize_func = lambda x: tokenizer(
        x["prompts"], truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    return dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=["response", "variation", "prompts"],
    )