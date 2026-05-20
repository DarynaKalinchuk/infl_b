"""
Evaluation metrics and post-processing of results.
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import json
import math



def run_benchmark_measures(influence, train_dataset, validation_dataset, metrics_path, time_elapsed):
    has_subvariation = (
        "subvariation" in train_dataset.column_names
        and "subvariation" in validation_dataset.column_names
    )


    if influence.shape != (len(validation_dataset), len(train_dataset)):
        raise ValueError(
            f"Influence shape mismatch: expected {(len(validation_dataset), len(train_dataset))} "
            f"(num_val, num_train), got {influence.shape}."
        )

    # 700/2 in backdoor case
    cov_cnt = int(len(train_dataset) / len(set(train_dataset["variation"])))
    n = len(influence)

    overall = {
        "variation": {"acc": 0, "cov": 0},
    }

    if has_subvariation:
        overall["subvariation"] = {"acc": 0, "cov": 0}

    per_variation = defaultdict(lambda: {"count": 0, "acc": 0, "cov": 0})
    per_subvariation = (
        defaultdict(lambda: {"count": 0, "acc": 0, "cov": 0})
        if has_subvariation
        else None
    )

    for i in range(n):
        array = -(influence.loc[i].to_numpy())

        indices = np.argpartition(array, -cov_cnt)[-cov_cnt:]
        topk_indices = indices[np.argsort(array[indices])[::-1]]

        val_variation = validation_dataset["variation"][i]
        per_variation[val_variation]["count"] += 1

        if has_subvariation:
            val_subvariation = validation_dataset["subvariation"][i]
            per_subvariation[val_subvariation]["count"] += 1

        top1_idx = int(topk_indices[0])

        if train_dataset["variation"][top1_idx] == val_variation:
            overall["variation"]["acc"] += 1
            per_variation[val_variation]["acc"] += 1

        if has_subvariation:
            if train_dataset["subvariation"][top1_idx] == val_subvariation:
                overall["subvariation"]["acc"] += 1
                per_subvariation[val_subvariation]["acc"] += 1

        var_cov_hits = 0
        subvar_cov_hits = 0

        for ele in topk_indices:
            ele = int(ele)

            if train_dataset["variation"][ele] == val_variation:
                overall["variation"]["cov"] += 1
                var_cov_hits += 1

            if has_subvariation:
                if train_dataset["subvariation"][ele] == val_subvariation:
                    overall["subvariation"]["cov"] += 1
                    subvar_cov_hits += 1

        per_variation[val_variation]["cov"] += var_cov_hits

        if has_subvariation:
            per_subvariation[val_subvariation]["cov"] += subvar_cov_hits

    metrics = {
        "time_elapsed": time_elapsed,
        "overall": {
            "variation": {
                "accuracy": overall["variation"]["acc"] / n,
                "coverage": overall["variation"]["cov"] / (n * cov_cnt),
            }
        },
        "per_variation": {},
    }

    if has_subvariation:
        metrics["overall"]["subvariation"] = {
            "accuracy": overall["subvariation"]["acc"] / n,
            "coverage": overall["subvariation"]["cov"] / (n * cov_cnt),
        }
        metrics["per_subvariation"] = {}

    for var, vals in per_variation.items():
        count = vals["count"]
        metrics["per_variation"][str(var)] = {
            "num_samples": count,
            "accuracy": vals["acc"] / count if count > 0 else 0.0,
            "coverage": vals["cov"] / (count * cov_cnt) if count > 0 else 0.0,
        }

    if has_subvariation:
        for subvar, vals in per_subvariation.items():
            count = vals["count"]
            metrics["per_subvariation"][str(subvar)] = {
                "num_samples": count,
                "accuracy": vals["acc"] / count if count > 0 else 0.0,
                "coverage": vals["cov"] / (count * cov_cnt) if count > 0 else 0.0,
            }

    print("Variation Acc:", metrics["overall"]["variation"]["accuracy"])
    print("Variation Cover:", metrics["overall"]["variation"]["coverage"])

    if has_subvariation:
        print("Subvariation Acc:", metrics["overall"]["subvariation"]["accuracy"])
        print("Subvariation Cover:", metrics["overall"]["subvariation"]["coverage"])

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



def plot_all_acc_cov(results_dir="results", figsize_per_subplot=(8, 6)):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    files = [
        f for f in os.listdir(results_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(results_dir, f))
    ]

    if not files:
        raise ValueError(f"No .json files found in {results_dir}")

    files = sorted(files)
    plot_data = []
    max_rows = 0

    for filename in files:
        filepath = os.path.join(results_dir, filename)

        with open(filepath, "r") as f:
            metrics = json.load(f)

        row_labels = []
        values = []

        overall = metrics.get("overall", {})
        time_elapsed = metrics.get("time_elapsed", None)

        # -------------------------
        # 1) overall variation first
        # -------------------------
        if "variation" in overall:
            row_labels.append("overall variation")
            values.append([
                overall["variation"].get("accuracy", 0.0),
                overall["variation"].get("coverage", 0.0),
            ])

        # -------------------------
        # 2) all variations
        # -------------------------
        per_variation = metrics.get("per_variation", {})
        for key in sorted(per_variation.keys(), key=str):
            val = per_variation[key]
            row_labels.append(f"variation: {key}")
            values.append([
                val.get("accuracy", 0.0),
                val.get("coverage", 0.0),
            ])

        # -------------------------
        # 3) overall subvariation
        # -------------------------
        if "subvariation" in overall:
            row_labels.append("overall subvariation")
            values.append([
                overall["subvariation"].get("accuracy", 0.0),
                overall["subvariation"].get("coverage", 0.0),
            ])

        # -------------------------
        # 4) all subvariations
        # -------------------------
        per_subvariation = metrics.get("per_subvariation", {})
        for key in sorted(per_subvariation.keys(), key=str):
            val = per_subvariation[key]
            row_labels.append(f"subvariation: {key}")
            values.append([
                val.get("accuracy", 0.0),
                val.get("coverage", 0.0),
            ])

        if values:
            values = np.array(values)
            plot_data.append((filename, row_labels, values, time_elapsed))
            max_rows = max(max_rows, len(row_labels))

    if not plot_data:
        raise ValueError(f"No plottable metrics found in JSON files in {results_dir}")

    n = len(plot_data)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig_width = figsize_per_subplot[0] * cols
    fig_height = max(figsize_per_subplot[1] * rows, 0.35 * max_rows * rows + 1.5)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    im = None

    for ax, (filename, row_labels, values, time_elapsed) in zip(axes, plot_data):
        im = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Accuracy", "Coverage"])

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        title = filename.replace(".json", "").replace("_", " ")

        if time_elapsed is not None:
            title += f"\nTime elapsed: {time_elapsed:.2f}s"

        ax.set_title(title, fontsize=10)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j] * 100:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    for ax in axes[len(plot_data):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes[:len(plot_data)],
            orientation="horizontal",
            fraction=0.04,
            pad=0.08
        )

        cbar.set_label("Rate")

        ticks = np.linspace(0, 1, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t * 100:.0f}%" for t in ticks])

    plt.savefig(
        os.path.join(results_dir, "acc_cov_all_in_one.png"),
        bbox_inches="tight"
    )

    plt.close(fig)