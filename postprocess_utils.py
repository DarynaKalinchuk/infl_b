"""
Evaluation metrics and post-processing of results.
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import json
import math
from sklearn.metrics import roc_auc_score
from matplotlib import cm
from matplotlib.colors import Normalize



def run_benchmark_measures(
    influence,
    train_dataset,
    validation_dataset,
    metrics_path,
    time_elapsed,
):
    fields = ["variation"]

    if (
        "subvariation" in train_dataset.column_names
        and "subvariation" in validation_dataset.column_names
    ):
        fields.append("subvariation")

    if influence.shape != (len(validation_dataset), len(train_dataset)):
        raise ValueError(
            f"Expected shape {(len(validation_dataset), len(train_dataset))}, "
            f"got {influence.shape}"
        )

    cov_cnt = len(train_dataset) // len(set(train_dataset["variation"]))
    n = len(validation_dataset)

    overall = {
    field: {"acc": 0, "cov": 0, "auc": 0}
    for field in fields
    }

    per_field = {
        field: defaultdict(
            lambda: {"count": 0, "acc": 0, "cov": 0, "auc": 0}
        )
        for field in fields
    }

    for i in range(n):
        scores = -influence.loc[i].to_numpy()

        indices = np.argpartition(scores, -cov_cnt)[-cov_cnt:]
        topk = indices[np.argsort(scores[indices])[::-1]]

        top1 = int(topk[0])

        for field in fields:
            val_label = validation_dataset[field][i]
        
            # Accuracy, Coverage
            stats = per_field[field][val_label]
            stats["count"] += 1

            if train_dataset[field][top1] == val_label:
                overall[field]["acc"] += 1
                stats["acc"] += 1

            cov_hits = sum(
                train_dataset[field][int(idx)] == val_label
                for idx in topk
            )

            overall[field]["cov"] += cov_hits
            stats["cov"] += cov_hits

            # AUC
            labels = (
                np.array(train_dataset[field]) == val_label
            ).astype(int)

            auc = roc_auc_score(labels, scores)

            overall[field]["auc"] += auc
            stats["auc"] += auc

    metrics = {
        "time_elapsed": time_elapsed,
        "overall": {},
    }

    for field in fields:
        metrics["overall"][field] = {
            "accuracy": overall[field]["acc"] / n,
            "coverage": overall[field]["cov"] / (n * cov_cnt),
            "auc": overall[field]["auc"] / n,
        }

        metrics[f"per_{field}"] = {
            str(label): {
                "num_samples": vals["count"],
                "accuracy": vals["acc"] / vals["count"],
                "coverage": vals["cov"] / (vals["count"] * cov_cnt),
                "auc": vals["auc"] / vals["count"],
            }
            for label, vals in per_field[field].items()
        }

        print(f"{field.title()} Acc:", metrics["overall"][field]["accuracy"])
        print(f"{field.title()} Cover:", metrics["overall"][field]["coverage"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_all_acc_cov(results_dir="results", figsize_scale=0.55, show=True):

    if not os.path.exists(results_dir):
        raise FileNotFoundError(
            f"Directory not found: {results_dir}"
        )

    files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
    )

    grouped = defaultdict(list)

    for file in files:
        stem = file.removesuffix(".json")
        parts = stem.split("_")

        group = (
            "_".join(parts[:2])
            if len(parts) >= 2
            else stem
        )

        exp = "_".join(parts[2:]) or "default"

        with open(os.path.join(results_dir, file)) as f:
            grouped[group].append((exp, json.load(f)))

    metric_names = ["accuracy", "coverage", "auc", "time"]
    metric_labels = ["Accuracy", "Coverage", "AUC", "Runtime"]
    fields = ["variation", "subvariation"]

    figures = {}

    for group_key, experiments in grouped.items():

        variation_names = sorted({
            f"overall {f}"
            for _, m in experiments
            for f in fields
            if f in m.get("overall", {})
        } | {
            f"{f}: {k}"
            for _, m in experiments
            for f in fields
            for k in m.get(f"per_{f}", {})
        })

        experiment_labels = []
        variation_labels = []

        for exp, _ in experiments:
            for i, variation in enumerate(variation_names):
                experiment_labels.append(
                    exp if i == 0 else ""
                )
                variation_labels.append(variation)

        row_names = [
            [exp_label, var_label]
            for exp_label, var_label in zip(
                experiment_labels,
                variation_labels
            )
        ]

        col_names = (
            ["Method", "Variation"]
            + metric_labels
        )

        values = np.full(
            (len(experiment_labels), len(metric_labels)),
            np.nan
        )

        for exp_idx, (exp, metrics) in enumerate(experiments):

            variation_map = {
                **{
                    f"overall {f}": metrics["overall"][f]
                    for f in fields
                    if f in metrics.get("overall", {})
                },
                **{
                    f"{f}: {k}": v
                    for f in fields
                    for k, v in metrics.get(f"per_{f}", {}).items()
                },
            }

            for var_idx, variation in enumerate(variation_names):

                row_idx = (
                    exp_idx * len(variation_names)
                    + var_idx
                )

                if variation not in variation_map:
                    continue

                vals = variation_map[variation]

                values[row_idx, 0] = vals.get(
                    "accuracy",
                    np.nan
                )

                values[row_idx, 1] = vals.get(
                    "coverage",
                    np.nan
                )

                values[row_idx, 2] = vals.get(
                    "auc",
                    np.nan
                )

                if variation.startswith("overall"):
                    values[row_idx, 3] = metrics.get(
                        "time_elapsed",
                        np.nan
                    )

        metric_norm = Normalize(0, 1)

        finite_times = values[:, 3][
            np.isfinite(values[:, 3])
        ]

        time_norm = Normalize(
            finite_times.min()
            if len(finite_times)
            else 0,
            finite_times.max()
            if len(finite_times)
            else 1,
        )

        cmap = cm.get_cmap("RdYlGn")

        cell_colours = [
            [
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                *[
                    (1, 1, 1, 1)
                    if np.isnan(v)
                    else cmap(
                        time_norm(v)
                        if c == 3
                        else metric_norm(v)
                    )
                    for c, v in enumerate(row)
                ]
            ]
            for row in values
        ]

        table_text = [
            [
                exp_label,
                var_label,
                *[
                    ""
                    if np.isnan(v)
                    else f"{v:.2f}s"
                    if c == 3
                    else f"{v * 100:.2f}%"
                    for c, v in enumerate(row)
                ]
            ]
            for exp_label, var_label, row in zip(
                experiment_labels,
                variation_labels,
                values
            )
        ]

        fig, ax = plt.subplots(
            figsize=(
                max(10, len(col_names) * 2),
                max(
                    4,
                    len(experiment_labels)
                    * figsize_scale + 2
                ),
            )
        )

        ax.axis("off")

        table = ax.table(
            cellText=table_text,
            cellColours=cell_colours,
            colLabels=col_names,
            cellLoc="center",
            loc="center",
        )

        for row in range(1, len(experiment_labels)):
            if experiment_labels[row] == "":
                table[(row + 1, 0)].visible_edges = "LR"

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)

        ax.set_title(
            group_key.replace("_", " "),
            fontsize=14,
            pad=20,
        )

        plt.tight_layout()

        output_path = os.path.join(
            results_dir,
            f"{group_key}_metrics_table.png"
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
            dpi=300
        )
