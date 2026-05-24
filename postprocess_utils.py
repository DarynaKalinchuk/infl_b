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
    field = "variation"

    if influence.shape != (len(validation_dataset), len(train_dataset)):
        raise ValueError(
            f"Expected shape {(len(validation_dataset), len(train_dataset))}, "
            f"got {influence.shape}"
        )

    cov_cnt = len(train_dataset) // len(set(train_dataset["variation"]))
    n = len(validation_dataset)

    overall = {"acc": 0, "cov": 0, "auc": 0, "range_ratio": 0}

    per_variation = defaultdict(
        lambda: {
            "count": 0,
            "acc": 0,
            "cov": 0,
            "auc": 0,
            "range_ratio": 0,
        }
    )

    for i in range(n):
        scores = -influence.loc[i].to_numpy()

        indices = np.argpartition(scores, -cov_cnt)[-cov_cnt:]
        topk = indices[np.argsort(scores[indices])[::-1]]
        top1 = int(topk[0])

        val_label = validation_dataset[field][i]

        labels = (np.array(train_dataset[field]) == val_label).astype(int)
        relevant_scores = scores[labels == 1]

        all_range = np.max(scores) - np.min(scores)
        rel_range = np.max(relevant_scores) - np.min(relevant_scores)
        range_ratio = rel_range / all_range if all_range > 0 else 0.0

        stats = per_variation[val_label]
        stats["count"] += 1

        if train_dataset[field][top1] == val_label:
            overall["acc"] += 1
            stats["acc"] += 1

        cov_hits = sum(
            train_dataset[field][int(idx)] == val_label
            for idx in topk
        )

        overall["cov"] += cov_hits
        stats["cov"] += cov_hits

        auc = roc_auc_score(labels, scores)
        overall["auc"] += auc
        stats["auc"] += auc

        overall["range_ratio"] += range_ratio
        stats["range_ratio"] += range_ratio

    metrics = {
        "time_elapsed": time_elapsed,
        "overall": {
            "variation": {
                "accuracy": overall["acc"] / n,
                "coverage": overall["cov"] / (n * cov_cnt),
                "auc": overall["auc"] / n,
                "range_ratio": overall["range_ratio"] / n,
            }
        },
        "per_variation": {
            str(label): {
                "num_samples": vals["count"],
                "accuracy": vals["acc"] / vals["count"],
                "coverage": vals["cov"] / (vals["count"] * cov_cnt),
                "auc": vals["auc"] / vals["count"],
                "range_ratio": vals["range_ratio"] / vals["count"],
            }
            for label, vals in per_variation.items()
        },
    }

    print("Accuracy:", metrics["overall"]["variation"]["accuracy"])
    print("Coverage:", metrics["overall"]["variation"]["coverage"])
    print("Range Ratio:", metrics["overall"]["variation"]["range_ratio"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def generate_table_metrics(results_dir="results", figsize_scale=0.55, show=True):

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    files = sorted(
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
    )

    grouped = defaultdict(list)

    for file in files:
        stem = file.removesuffix(".json")
        parts = stem.split("_")

        group = "_".join(parts[:2]) if len(parts) >= 2 else stem
        exp = parts[2]

        with open(os.path.join(results_dir, file)) as f:
            grouped[group].append((exp, json.load(f)))

    metric_names = ["accuracy", "coverage", "auc", "range_ratio", "time"]
    metric_labels = ["Accuracy", "Coverage", "AUC", "Range Ratio", "Runtime"]

    figures = {}

    for group_key, experiments in grouped.items():

        variation_names = sorted({
            "overall variation"
            for _, m in experiments
            if "variation" in m.get("overall", {})
        } | {
            f"variation: {k}"
            for _, m in experiments
            for k in m.get("per_variation", {})
        })

        experiment_labels = []
        variation_labels = []

        for exp, _ in experiments:
            for i, variation in enumerate(variation_names):
                experiment_labels.append(exp if i == 0 else "")
                variation_labels.append(variation)

        col_names = ["Method", "Variation"] + metric_labels

        values = np.full(
            (len(experiment_labels), len(metric_labels)),
            np.nan
        )

        for exp_idx, (exp, metrics) in enumerate(experiments):

            variation_map = {
                "overall variation": metrics["overall"]["variation"],
                **{
                    f"variation: {k}": v
                    for k, v in metrics.get("per_variation", {}).items()
                },
            }

            for var_idx, variation in enumerate(variation_names):
                row_idx = exp_idx * len(variation_names) + var_idx

                if variation not in variation_map:
                    continue

                vals = variation_map[variation]

                values[row_idx, 0] = vals.get("accuracy", np.nan)
                values[row_idx, 1] = vals.get("coverage", np.nan)
                values[row_idx, 2] = vals.get("auc", np.nan)
                values[row_idx, 3] = vals.get("range_ratio", np.nan)

                if variation.startswith("overall"):
                    values[row_idx, 4] = metrics.get("time_elapsed", np.nan)

        cmap = cm.get_cmap("RdYlGn")
        reverse_cmap = cm.get_cmap("RdYlGn_r")

        col_norms = []

        for c in range(values.shape[1]):
            finite = values[:, c][np.isfinite(values[:, c])]

            if len(finite) and finite.min() != finite.max():
                col_norms.append(Normalize(finite.min(), finite.max()))
            else:
                col_norms.append(Normalize(0, 1))

        cell_colours = [
            [
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                *[
                    (1, 1, 1, 1)
                    if np.isnan(v)
                    else (
                        reverse_cmap(col_norms[c](v))
                        if c in [3, 4]  # Range Ratio + Runtime: lower is better
                        else cmap(col_norms[c](v))
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
                    if c == 4
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
                max(4, len(experiment_labels) * figsize_scale + 2),
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

        figures[group_key] = fig

        if show:
            plt.show()
        else:
            plt.close(fig)

    return figures
