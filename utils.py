import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
import json
import math

def get_preprocessed_dataset(tokenizer, dataset, chat_template, max_length):
    def apply_prompt_template(sample):
        return {
            'text': chat_template.format(prompt=sample['prompts'], response=sample['response'])
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenized_dataset(text):
        input_text = text['text']
        tokenized_output = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
        tokenized_output['labels'] = tokenized_output['input_ids'].copy()
        return tokenized_output

    return dataset.map(tokenized_dataset, batched=True, remove_columns=['text'])

def collect_gradient(model, tokenizer, tokenized_tr, tokenized_val):

    
    collate_fn = lambda x: tokenizer.pad(x, padding="longest", return_tensors="pt")
    train_dataloader_stochastic = DataLoader(tokenized_tr, 
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              batch_size=1)
    val_dataloader_stochastic = DataLoader(tokenized_val, 
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              batch_size=1)

    model.eval()
    tr_grad_dict = {}
    for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
        model.zero_grad()
        batch['labels'] = batch['input_ids']
        batch.to('cuda')
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
            
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        tr_grad_dict[step] = grad_dict
        del grad_dict
            
    val_grad_dict = {}
    for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
        model.zero_grad()
        batch['labels'] = batch['input_ids']
        batch.to('cuda')
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
            
        grad_dict = {}
        for k, v in model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k] = v.grad.cpu()
            elif 'lora_B' in k:
                grad_dict[k] = v.grad.cpu().T
            else: pass
        val_grad_dict[step] = grad_dict    
        del grad_dict
            
    return tr_grad_dict, val_grad_dict


def template_setting(model_n):
    if model_n == 'Llama':
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        chat_template = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            "{prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "{response}"
        )
    elif model_n == 'Qwen0.5':
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        chat_template = (
            "<|im_start|>user\n"
            "{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{response}<|im_end|>"
        )
    
    elif model_n == 'Qwen1.5':
        model_name = "Qwen/Qwen2-1.5B-Instruct"
        chat_template = (
            "<|im_start|>user\n"
            "{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{response}<|im_end|>"
        )

    elif model_n == "Olmo":
        model_name = "allenai/OLMo-2-0425-1B-SFT"
        chat_template = (
            "<|user|>\n"
            "{prompt}\n"
            "<|assistant|>\n"
            "{response}<|endoftext|>"
        )

    return model_name, chat_template



"""
Required set-up for influence function computation.
Prepares models for running EK-FAC and defines the language modeling task
(in our case, cross-entropy training and query loss).
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]

class LanguageModelingTask(Task):
    def __init__(
        self, 
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        num_layers = self.config['model']['num_layers']

        if self.config['model']['family'] in ["llama"]:

            for i in range(num_layers):
                    total_modules.append(f"model.layers.{i}.mlp.gate_proj")
                    total_modules.append(f"model.layers.{i}.mlp.up_proj")
                    total_modules.append(f"model.layers.{i}.mlp.down_proj")

                    total_modules.append(f"model.layers.{i}.self_attn.q_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.k_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.v_proj")
                    total_modules.append(f"model.layers.{i}.self_attn.o_proj")  

        # elif 
        else:
            raise NotImplementedError(
                f"Model family {self.config['model']['family']} not supported for influence tracking."
                " Please add the required modules to the `get_influence_tracked_modules` method in `utils/tasks.py`."
            )

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]

