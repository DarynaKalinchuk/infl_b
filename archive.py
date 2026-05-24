
def hvp_estimate(
    model,
    batch,
    vector,
    device="cuda"
):
    """
    Exact Hessian-vector product for LoRA params.

    Args:
        model: PEFT / LoRA model
        batch: tokenized batch
        vector: flattened vector matching LoRA params
    Returns:
        flattened Hv
    """

    model.zero_grad()

    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = batch["input_ids"]

    outputs = model(**batch)
    loss = outputs.loss

    params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ("lora_" in n)
    ]

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True
    )

    flat_grads = torch.cat([
        g.reshape(-1)
        for g in grads
    ])

    grad_dot_vec = torch.sum(flat_grads * vector)

    hv = torch.autograd.grad(
        grad_dot_vec,
        params,
        retain_graph=False
    )

    flat_hv = torch.cat([
        h.reshape(-1)
        for h in hv
    ])

    return flat_hv.detach()


########## EKFAC STUFF #############################
# influence.py :



if args.hvp_cal == "ekfac":
    # these are from sbatch file
    HVP_METHOD="ekfac"
    INF_ARGS="lambda_const_param=0.01,n_iteration=10,alpha_const=1."

    
    model = PeftModel.from_pretrained(
        base_model,
        "lora_adapter/" + core_path
    )

            
    config = {
        "model": {
            "family": "olmo",
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


# utils




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

        if self.config['model']['family'] in ["olmo"]:

            for i in range(num_layers):

                total_modules.append(
                    f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default"
                )
                total_modules.append(
                    f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.default"
                )

                total_modules.append(
                    f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.default"
                )
                total_modules.append(
                    f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default"
                )

        # elif 
        else:
            raise NotImplementedError(
                f"Model family {self.config['model']['family']} not supported for influence tracking."
                " Please add the required modules to the `get_influence_tracked_modules` method in `utils/tasks.py`."
            )

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]



# inf_estimation_methods


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




