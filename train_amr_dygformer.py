#!/usr/bin/env python3
"""
train_amr_dygformer.py

Trainer for temporal AMR graph prediction using AMRDyGFormer.

Data flow
- Trains on temporal graph windows built from a user-specified .pt graph folder.
- Optionally evaluates on an explicitly provided external test folder or, when
  not provided, on an automatically detected external test folder located beside
  the training data folder.
- Uses TemporalGraphDataset to construct length-T windows with configurable
  sliding step.

Model and training
- Supports classification and regression tasks resolved from tasks.py.
- Builds an AMRDyGFormer model with configurable GraphSAGE and transformer depth.
- Supports standard training or evaluation-only execution from a saved checkpoint.
- Performs validation splitting either at trajectory level using sim_id metadata
  or, when necessary, by fallback random window split.

Sampling and graph processing
- Supports legacy edge-thinning through max_neighbors.
- Supports neighborhood sampling with PyG NeighborLoader using configurable:
    --neighbor_sampling
    --num_neighbors
    --seed_count
    --seed_strategy
    --seed_batch_size
    --max_sub_batches
- When task hyperparameters are enabled, command-line sampling settings take
  precedence over task-level defaults.

Metadata and temporal consistency
- Prefers Data.sim_id and Data.day metadata when constructing temporal windows.
- Can require .pt metadata and optionally fail on non-contiguous windows.
- Uses trajectory identifiers to perform trajectory-level train/validation splits.

Outputs
- Writes training curves, validation/test performance plots, attention heatmaps,
  metrics summaries, and the trained model checkpoint to the chosen output directory.
- Emits progress markers for integration with external interfaces.

Typical output files
- loss_curves.png
- confusion_matrix.png
- roc_curve.png
- confusion_matrix_test.png
- roc_curve_test.png
- attention_heatmap.png
- attention_heatmap.csv
- attention_heatmap_test.png
- attention_heatmap_test.csv
- metrics_summary.json
- metrics_summary.txt
- trained_model.pt
"""

import argparse
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import csv

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Sampler

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from temporal_graph_dataset import TemporalGraphDataset, collate_temporal_graph_batch
from models_amr import AMRDyGFormer
from tasks import TASK_REGISTRY, BaseTask, get_task

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

# =============================================================================
# CLI helpers
# =============================================================================

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_num_neighbors(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def configure_cpu_threads(cpu_threads: int) -> int:
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass
    try:
        interop_threads = max(1, min(cpu_threads, 4))
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass
    return cpu_threads


# =============================================================================
# Legacy edge-thinning (kept)
# =============================================================================

def _subsample_edges_per_graph(g, max_neighbors: int):
    if max_neighbors <= 0:
        return g

    edge_index = g.edge_index
    edge_attr = getattr(g, "edge_attr", None)
    if edge_index.numel() == 0:
        return g

    dst = edge_index[1].cpu().numpy()
    num_nodes = g.num_nodes
    indices_by_dst = [[] for _ in range(num_nodes)]
    for e_idx, j in enumerate(dst):
        if 0 <= j < num_nodes:
            indices_by_dst[j].append(e_idx)

    keep: List[int] = []
    for group in indices_by_dst:
        if len(group) <= max_neighbors:
            keep.extend(group)
        else:
            choice = np.random.choice(group, size=max_neighbors, replace=False)
            keep.extend(choice.tolist())

    keep = sorted(set(keep))
    if len(keep) == 0:
        return g

    g.edge_index = edge_index[:, keep]
    if edge_attr is not None:
        g.edge_attr = edge_attr[keep]
    return g


def subsample_neighbors_in_batches(batched_graphs, max_neighbors: int):
    if max_neighbors <= 0:
        return batched_graphs
    out = []
    for g in batched_graphs:
        g_cpu = g.cpu()
        out.append(_subsample_edges_per_graph(g_cpu, max_neighbors))
    return out


# =============================================================================
# True GraphSAGE neighbor sampling (NeighborLoader)
# =============================================================================

def _select_seed_nodes(num_nodes: int, seed_count: int, seed_strategy: str) -> torch.Tensor:
    if num_nodes <= 0:
        return torch.empty((0,), dtype=torch.long)

    strat = str(seed_strategy).lower().strip()
    if strat == "all" or seed_count <= 0 or seed_count >= num_nodes:
        return torch.arange(num_nodes, dtype=torch.long)

    perm = torch.randperm(num_nodes)
    return perm[:seed_count].to(torch.long)

def _sanitize_graph_for_neighbor_loader(g_data) -> Data:
    """
    Build a minimal PyG Data object compatible with NeighborLoader.

    Keep only attributes needed by the GNN forward under sampled training:
      - x
      - edge_index
      - edge_attr (if present)
      - node_id (if present and tensor-like)

    Drop any auxiliary Python-list metadata that can cause NeighborLoader's
    internal filtering/index_select path to fail.
    """
    x = g_data.x
    edge_index = g_data.edge_index
    edge_attr = getattr(g_data, "edge_attr", None)

    clean = Data(x=x, edge_index=edge_index)

    if edge_attr is not None:
        clean.edge_attr = edge_attr

    node_id = getattr(g_data, "node_id", None)
    if torch.is_tensor(node_id):
        clean.node_id = node_id
    elif node_id is not None:
        try:
            clean.node_id = torch.as_tensor(node_id, dtype=torch.long)
        except Exception:
            pass

    return clean


def _embed_one_graph_with_neighbor_sampling(
    model: AMRDyGFormer,
    g_data,
    device: torch.device,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
) -> torch.Tensor:
    num_nodes = int(g_data.num_nodes)
    seeds = _select_seed_nodes(num_nodes=num_nodes, seed_count=seed_count, seed_strategy=seed_strategy)

    if seeds.numel() == 0:
        return torch.zeros((model.hidden_channels,), device=device, dtype=torch.float32)

    g_cpu = _sanitize_graph_for_neighbor_loader(g_data.cpu())
    seeds = seeds.cpu()

    bs = int(min(seed_batch_size, seeds.numel()))
    loader = NeighborLoader(
        g_cpu,
        input_nodes=seeds,
        num_neighbors=num_neighbors,
        batch_size=bs,
        shuffle=True,
    )

    pooled_list: List[torch.Tensor] = []
    for b_idx, sub in enumerate(loader):
        if max_sub_batches > 0 and b_idx >= max_sub_batches:
            break

        sub = sub.to(device)
        x = sub.x
        edge_index = sub.edge_index
        edge_attr = getattr(sub, "edge_attr", None)

        batch_vec = torch.zeros((x.size(0),), device=device, dtype=torch.long)
        pooled = model.gnn(x, edge_index, edge_attr, batch_vec)
        pooled_list.append(pooled.squeeze(0))

    if len(pooled_list) == 0:
        return torch.zeros((model.hidden_channels,), device=device, dtype=torch.float32)

    return torch.stack(pooled_list, dim=0).mean(dim=0)


def build_day_embeddings_neighbor_sampling(
    model: AMRDyGFormer,
    batched_graphs: List,
    device: torch.device,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
) -> torch.Tensor:
    day_embs: List[torch.Tensor] = []
    for g_batch in batched_graphs:
        data_list = g_batch.to_data_list()
        emb_rows: List[torch.Tensor] = []
        for g_data in data_list:
            emb_i = _embed_one_graph_with_neighbor_sampling(
                model=model,
                g_data=g_data,
                device=device,
                num_neighbors=num_neighbors,
                seed_count=seed_count,
                seed_strategy=seed_strategy,
                max_sub_batches=max_sub_batches,
                seed_batch_size=seed_batch_size,
            )
            emb_rows.append(emb_i)
        day_emb = torch.stack(emb_rows, dim=0)
        day_embs.append(day_emb)

    H = torch.stack(day_embs, dim=1)
    return H


# =============================================================================
# ROC / confusion-matrix plotting helpers
# =============================================================================

def _get_class_names(task, n_classes: int):
    names = getattr(task, "class_names", None)
    if isinstance(names, (list, tuple)) and len(names) >= n_classes:
        return list(names)[:n_classes]
    return [f"class {i}" for i in range(n_classes)]


def plot_confusion_matrix_annotated(
    cm: np.ndarray,
    out_path: str,
    title: str,
    class_names: Optional[List[str]] = None,
):
    cm = np.asarray(cm)
    n_classes = int(cm.shape[0])

    if class_names is None or len(class_names) < n_classes:
        class_names = [str(i) for i in range(n_classes)]
    else:
        class_names = [str(x) for x in class_names[:n_classes]]

    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    #fig_h = 5.0 if n_classes <= 4 else min(12.0, 4.0 + 0.45 * n_classes)
    #fig_w = 6.5 if n_classes <= 4 else min(12.0, 5.0 + 0.45 * n_classes)
    
    fig_h = 6.5 if n_classes <= 4 else min(14.0, 5.0 + 0.55 * n_classes)
    fig_w = 8.0 if n_classes <= 4 else min(14.0, 6.0 + 0.55 * n_classes)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    #im = ax.imshow(norm_cm, cmap="cividis", vmin=0.0, vmax=1.0)
    im = ax.imshow(norm_cm, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.set_xticks(list(range(n_classes)))
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=13)
    ax.set_yticks(list(range(n_classes)))
    ax.set_yticklabels(class_names, fontsize=13)

    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            pct = norm_cm[i, j] * 100.0
            txt = f"{int(cm[i, j])}\n{pct:.1f}%"
            color = "white" if norm_cm[i, j] >= threshold else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=color,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_roc_curves(y_true, probs, n_classes: int, out_path: str, title: str, class_names=None):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)

    plt.figure(figsize=(7.8, 5.8))
    plotted_any = False

    if y_true.size == 0:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.6, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.text(0.5, 0.5, "No samples to plot", ha="center", va="center")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.28)
        plt.tight_layout()
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close()
        return

    if n_classes == 2:
        y_score = probs[:, 1]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2.8, label=f"Model (AUC = {roc_auc:.3f})")
            plt.fill_between(fpr, tpr, alpha=0.15)
            plotted_any = True
        except ValueError as e:
            print(f"⚠️ ROC curve skipped ({title}): {e}")

    else:
        classes = list(range(n_classes))
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)

        if class_names is None:
            class_names = [f"class {i}" for i in classes]

        for i in classes:
            positives = float(y_bin[:, i].sum())
            if positives == 0.0 or positives == float(len(y_bin)):
                print(f"⚠️ ROC for '{class_names[i]}' skipped ({title}): class not present in this split.")
                continue
            try:
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i], drop_intermediate=False)
                auc_i = auc(fpr_i, tpr_i)
                plt.plot(fpr_i, tpr_i, linewidth=2.2, label=f"{class_names[i]} (AUC = {auc_i:.3f})")
                plotted_any = True
            except ValueError as e:
                print(f"⚠️ ROC for '{class_names[i]}' skipped ({title}): {e}")

        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs[:, :n_classes].ravel(), drop_intermediate=False)
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(
                fpr_micro,
                tpr_micro,
                linestyle=":",
                linewidth=2.4,
                label=f"Micro-average (AUC = {auc_micro:.3f})",
            )
            plotted_any = True
        except Exception as e:
            print(f"⚠️ Micro-average ROC skipped ({title}): {e}")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if plotted_any:
        plt.legend(
        loc="lower right",
        fontsize=15,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="black",
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.0,
    )
    else:
        plt.text(0.5, 0.5, "ROC undefined (single-class labels)", ha="center", va="center")
    plt.grid(alpha=0.28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _classification_split_summary(
    task: BaseTask,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probs = np.asarray(probs)
    preds = np.asarray(preds).astype(int).reshape(-1)

    from sklearn.metrics import accuracy_score

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        preds,
        labels=list(range(task.out_dim)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, preds, labels=list(range(task.out_dim)))
    unique_classes = sorted(set(int(x) for x in y_true.tolist()))

    roc_details: Dict[str, Any] = {}
    try:
        if len(unique_classes) >= 2:
            if task.out_dim == 2:
                roc_details["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                roc_details["roc_auc_macro_ovr"] = float(
                    roc_auc_score(
                        y_true,
                        probs[:, : task.out_dim],
                        multi_class="ovr",
                        average="macro",
                    )
                )
    except Exception:
        pass

    per_class = []
    for i in range(task.out_dim):
        per_class.append(
            {
                "class_index": int(i),
                "class_name": str(class_names[i]) if i < len(class_names) else f"class {i}",
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support_per_class[i]),
            }
        )

    out = {
        "n_samples": int(y_true.shape[0]),
        "class_names": [str(x) for x in class_names],
        "metrics": {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision_macro": float(np.mean(precision_per_class)) if precision_per_class.size > 0 else 0.0,
            "recall_macro": float(np.mean(recall_per_class)) if recall_per_class.size > 0 else 0.0,
            "f1_macro": float(np.mean(f1_per_class)) if f1_per_class.size > 0 else 0.0,
            "precision_weighted": float(
                np.average(precision_per_class, weights=support_per_class)
            ) if support_per_class.sum() > 0 else 0.0,
            "recall_weighted": float(
                np.average(recall_per_class, weights=support_per_class)
            ) if support_per_class.sum() > 0 else 0.0,
            "f1_weighted": float(
                np.average(f1_per_class, weights=support_per_class)
            ) if support_per_class.sum() > 0 else 0.0,
            **roc_details,
        },
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "y_true": y_true.tolist(),
        "y_pred": preds.tolist(),
        "probabilities": probs.tolist(),
    }
    return out


def _regression_split_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_pred - y_true) ** 2)) if y_true.size > 0 else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true))) if y_true.size > 0 else 0.0
    return {
        "n_samples": int(y_true.shape[0]),
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        },
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def _write_run_summary_files(out_dir: str, summary: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "metrics_summary.json")
    txt_path = os.path.join(out_dir, "metrics_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, indent=2)

    lines: List[str] = []
    lines.append("RUN METRICS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"task: {summary.get('task', '')}")
    lines.append(f"is_classification: {summary.get('is_classification', '')}")
    lines.append(f"output_dir: {summary.get('output_dir', '')}")
    lines.append("")

    config = summary.get("config", {})
    if isinstance(config, dict) and config:
        lines.append("CONFIG")
        lines.append("-" * 80)
        for k in sorted(config.keys()):
            lines.append(f"{k}: {config[k]}")
        lines.append("")

    for split_name in ("validation", "test"):
        split = summary.get(split_name, None)
        if not isinstance(split, dict) or not split:
            continue

        lines.append(split_name.upper())
        lines.append("-" * 80)
        lines.append(f"n_samples: {split.get('n_samples', 0)}")

        metrics = split.get("metrics", {})
        if isinstance(metrics, dict):
            for k in sorted(metrics.keys()):
                v = metrics[k]
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.10f}")
                else:
                    lines.append(f"{k}: {v}")

        cm = split.get("confusion_matrix", None)
        if cm is not None:
            lines.append("confusion_matrix:")
            for row in cm:
                lines.append("  " + " ".join(str(int(x)) for x in row))

        per_class = split.get("per_class", None)
        if isinstance(per_class, list) and per_class:
            lines.append("per_class:")
            for row in per_class:
                cname = row.get("class_name", row.get("class_index", ""))
                lines.append(
                    "  "
                    f"{cname}: precision={float(row.get('precision', 0.0)):.10f}, "
                    f"recall={float(row.get('recall', 0.0)):.10f}, "
                    f"f1={float(row.get('f1', 0.0)):.10f}, "
                    f"support={int(row.get('support', 0))}"
                )
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# =============================================================================
# Attention heatmaps (RESTORED)
# =============================================================================

def _load_node_vocab_inv(folder: str, filename: str = "node_vocab.json") -> Optional[List[str]]:
    """
    Load node_vocab_inv list where index == node_id.
    Expected JSON:
      { "node_vocab_inv": ["name0","name1", ...] }
    """
    path = os.path.join(os.path.abspath(folder), filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        inv = obj.get("node_vocab_inv", None)
        if isinstance(inv, list) and len(inv) > 0:
            return [str(x) for x in inv]
    except Exception:
        return None
    return None


def save_attention_heatmap(
    model: AMRDyGFormer,
    task: BaseTask,
    loader,
    device: torch.device,
    max_neighbors: int,
    neighbor_sampling: bool,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
    out_png: str,
    out_csv: str,
    split_name: str,
    node_vocab_inv: Optional[List[str]] = None,
    top_k: int = 50,
    rank_by: str = "abs_diff",
):
    """
    Save a 600dpi PNG + CSV attention heatmap relating node-attention to task performance.

    For classification:
      Groups are Correct vs Incorrect
    For regression:
      Groups are Low error vs High error using median(|err|) split.

    Ranking:
      - abs_diff: rank by |mean(attn_good) - mean(attn_bad)|
      - mean: rank by mean attention across both groups
    """
    model.eval()

    # Defensive defaults
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 50
    if top_k <= 0:
        top_k = 50
    rank_by = str(rank_by).strip().lower()
    if rank_by not in ("abs_diff", "mean"):
        rank_by = "abs_diff"

    # Under NeighborLoader we cannot align attention to full-graph node IDs consistently.
    if neighbor_sampling:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            f"Attention heatmap unavailable ({split_name}).\nDisable neighbor sampling to export node attention.",
            ha="center",
            va="center",
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("note\n")
            f.write("Attention heatmap unavailable under neighbor sampling\n")
        return

    # Per-sample storage:
    #   sample_attn[i] : dict(node_id -> mean attention across days)
    #   sample_perf[i] : classification correctness (1/0) OR regression abs error
    sample_attn: List[Dict[int, float]] = []
    sample_perf: List[float] = []

    with torch.no_grad():
        for batched_graphs, labels_dict in loader:
            batched_graphs = subsample_neighbors_in_batches(batched_graphs, max_neighbors)
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
            labels_dict = _move_labels_dict_to_device(labels_dict, device)

            # Predictions
            action_features = _extract_action_features(batched_graphs_dev, labels_dict, device=device)
            state_summary_features = _extract_state_summary_features(batched_graphs_dev, labels_dict, device=device)
            y_hat = model(
                batched_graphs_dev,
                action_features=action_features,
                state_summary_features=state_summary_features,
            )

            # Compute per-day per-sample node attention maps: [T][B]{node_id:attn}
            per_day_maps: List[List[Dict[int, float]]] = []

            for g in batched_graphs_dev:
                x = g.x
                edge_index = g.edge_index
                edge_attr = getattr(g, "edge_attr", None)
                batch_vec = g.batch

                # Expected that gnn supports return_attention=True
                try:
                    _, attn_vec = model.gnn(x, edge_index, edge_attr, batch_vec, return_attention=True)
                except TypeError as e:
                    raise TypeError(
                        "models_amr.AMRDyGFormer.gnn must support return_attention=True to export heatmaps."
                    ) from e

                node_id_vec = getattr(g, "node_id", None)
                if node_id_vec is None:
                    node_id_vec = torch.arange(x.size(0), device=device, dtype=torch.long)

                num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
                day_list: List[Dict[int, float]] = []
                for i in range(num_graphs):
                    idx = (batch_vec == i).nonzero(as_tuple=False).view(-1)
                    if idx.numel() == 0:
                        day_list.append({})
                        continue

                    ids_i = node_id_vec[idx].detach().cpu().numpy().astype(int).tolist()
                    attn_i = attn_vec[idx].detach().cpu().numpy().astype(float).tolist()

                    d: Dict[int, float] = {}
                    c: Dict[int, int] = {}
                    for nid, a in zip(ids_i, attn_i):
                        if nid in d:
                            d[nid] += float(a)
                            c[nid] += 1
                        else:
                            d[nid] = float(a)
                            c[nid] = 1
                    for nid in list(d.keys()):
                        d[nid] /= float(c[nid])
                    day_list.append(d)

                per_day_maps.append(day_list)

            if len(per_day_maps) == 0:
                continue

            B = len(per_day_maps[0])
            T = len(per_day_maps)

            # Average per-node attention across days, per sample
            for i in range(B):
                maps_i = [per_day_maps[t][i] for t in range(T)]
                all_ids = set()
                for d in maps_i:
                    all_ids |= set(d.keys())

                if len(all_ids) == 0:
                    sample_attn.append({})
                    continue

                avg_map: Dict[int, float] = {}
                for nid in all_ids:
                    vals = [d.get(nid, np.nan) for d in maps_i]
                    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                    if len(vals) == 0:
                        continue
                    avg_map[int(nid)] = float(np.mean(vals))
                sample_attn.append(avg_map)

            # Performance signal aligned to B
            if task.is_classification:
                probs = F.softmax(y_hat, dim=1)
                pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
                y_true = task.get_targets(batched_graphs_dev, labels_dict).detach().cpu().numpy().astype(int).reshape(-1)
                m = min(B, len(pred), len(y_true))
                for j in range(m):
                    sample_perf.append(float(pred[j] == y_true[j]))
            else:
                y_true = task.get_targets(batched_graphs_dev, labels_dict).detach().cpu().numpy().reshape(-1)
                y_pred = y_hat.detach().cpu().numpy().reshape(-1)
                m = min(B, len(y_true), len(y_pred))
                for j in range(m):
                    sample_perf.append(float(abs(y_pred[j] - y_true[j])))

    # Align lengths
    m = min(len(sample_attn), len(sample_perf))
    sample_attn = sample_attn[:m]
    perf = np.asarray(sample_perf[:m], dtype=float)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    if m == 0:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, f"No samples to plot ({split_name}).", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("note\n")
            f.write("No samples to plot\n")
        return

    # Group masks and labels
    if task.is_classification:
        good_mask = perf >= 0.5
        bad_mask = ~good_mask
        col_labels = ["Correct", "Incorrect"]
        acc = float(np.mean(perf)) if perf.size else float("nan")
        title = f"Node attention vs performance ({split_name}) | acc={acc:.3f} | n={m}"
    else:
        thr = float(np.median(perf)) if perf.size else 0.0
        good_mask = perf <= thr
        bad_mask = perf > thr
        col_labels = ["Low error", "High error"]
        title = f"Node attention vs error ({split_name}) | median|err|={thr:.3g} | n={m}"

    def _agg(mask: np.ndarray) -> Dict[int, float]:
        sums: Dict[int, float] = {}
        cnts: Dict[int, int] = {}
        idxs = np.where(mask)[0].tolist()
        for k in idxs:
            d = sample_attn[k]
            for nid, a in d.items():
                sums[nid] = sums.get(nid, 0.0) + float(a)
                cnts[nid] = cnts.get(nid, 0) + 1
        out: Dict[int, float] = {}
        for nid, ssum in sums.items():
            c = cnts.get(nid, 0)
            if c > 0:
                out[nid] = ssum / float(c)
        return out

    mean_good = _agg(good_mask)
    mean_bad = _agg(bad_mask)

    node_ids = sorted(set(mean_good.keys()) | set(mean_bad.keys()))
    if len(node_ids) == 0:
        plt.figure(figsize=(6, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, f"No node IDs to plot ({split_name}).", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("note\n")
            f.write("No node IDs to plot\n")
        return

    rows: List[Tuple[int, float, float, float, float]] = []
    for nid in node_ids:
        a = float(mean_good.get(nid, np.nan))
        b = float(mean_bad.get(nid, np.nan))
        diff = a - b if (not np.isnan(a) and not np.isnan(b)) else np.nan
        mean_ = float(np.nanmean([a, b]))
        rows.append((int(nid), a, b, diff, mean_))

    if rank_by == "mean":
        rows.sort(key=lambda r: (-(r[4] if not np.isnan(r[4]) else -1e18), r[0]))
    else:
        rows.sort(key=lambda r: (-(abs(r[3]) if not np.isnan(r[3]) else -1e18), r[0]))

    rows = rows[: min(top_k, len(rows))]
    Hm = np.stack([[r[1], r[2]] for r in rows], axis=0)

    # Write CSV
    col1 = col_labels[0].replace(" ", "_")
    col2 = col_labels[1].replace(" ", "_")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"node_id,node_name,{col1},{col2},diff\n")
        for nid, a, b, diff, _ in rows:
            name = str(nid)
            if node_vocab_inv is not None and 0 <= nid < len(node_vocab_inv):
                name = str(node_vocab_inv[nid])
            f.write(f"{nid},{name},{a:.10f},{b:.10f},{diff:.10f}\n")

    # Plot heatmap
    fig_h = max(4.0, min(18.0, Hm.shape[0] / 10.0))
    plt.figure(figsize=(7.5, fig_h))
    plt.imshow(Hm, aspect="auto")
    plt.colorbar(label="Mean attention")
    plt.xticks(ticks=[0, 1], labels=col_labels)

    y_labels = []
    for nid, *_ in rows:
        if node_vocab_inv is not None and 0 <= nid < len(node_vocab_inv):
            y_labels.append(str(node_vocab_inv[nid]))
        else:
            y_labels.append(str(nid))

    plt.yticks(ticks=list(range(len(y_labels))), labels=y_labels, fontsize=7)
    plt.ylabel("Nodes (stable ID)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# Train/eval
# =============================================================================

def _move_labels_dict_to_device(
    labels_dict: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in labels_dict.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _merge_collected_labels(collected_labels: Dict[str, List[Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for k, values in collected_labels.items():
        if len(values) == 0:
            continue

        first = values[0]
        if torch.is_tensor(first):
            merged[k] = torch.cat(values, dim=0)
        else:
            flat: List[Any] = []
            for v in values:
                if isinstance(v, (list, tuple)):
                    flat.extend(list(v))
                else:
                    flat.append(v)
            merged[k] = flat
    return merged

def train_epoch(
    model,
    task,
    loader,
    optimizer,
    device,
    max_neighbors: int,
    neighbor_sampling: bool,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
    task_name: str,
    aux_policy_loss: bool,
    aux_policy_loss_weight: float,
    aux_policy_target_name: str,
    pairwise_policy_ranking_loss: bool,
    pairwise_policy_ranking_weight: float,
    pairwise_policy_margin: float,
    pairwise_policy_min_target_gap: float,
    oracle_best_action_loss: bool,
    oracle_best_action_loss_weight: float,
):
    model.train()
    total_loss = 0.0
    n = 0

    for batched_graphs, labels_dict in loader:
        if neighbor_sampling:
            H = build_day_embeddings_neighbor_sampling(
                model=model,
                batched_graphs=batched_graphs,
                device=device,
                num_neighbors=num_neighbors,
                seed_count=seed_count,
                seed_strategy=seed_strategy,
                max_sub_batches=max_sub_batches,
                seed_batch_size=seed_batch_size,
            )
            action_features = _extract_action_features(batched_graphs, labels_dict, device=device)
            state_summary_features = _extract_state_summary_features(batched_graphs, labels_dict, device=device)
            y_hat = model.forward_from_day_embeddings(
                H,
                action_features=action_features,
                state_summary_features=state_summary_features,
            )
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
        else:
            batched_graphs = subsample_neighbors_in_batches(batched_graphs, max_neighbors)
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
            action_features = _extract_action_features(batched_graphs_dev, labels_dict, device=device)
            state_summary_features = _extract_state_summary_features(batched_graphs_dev, labels_dict, device=device)
            y_hat = model(
                batched_graphs_dev,
                action_features=action_features,
                state_summary_features=state_summary_features,
            )

        labels_dict = _move_labels_dict_to_device(labels_dict, device)

        optimizer.zero_grad()
        loss = _compute_total_loss(
            model=model,
            task=task,
            y_hat=y_hat,
            batched_graphs_dev=batched_graphs_dev,
            labels_dict=labels_dict,
            task_name=task_name,
            aux_policy_loss=aux_policy_loss,
            aux_policy_loss_weight=aux_policy_loss_weight,
            aux_policy_target_name=aux_policy_target_name,
            pairwise_policy_ranking_loss=pairwise_policy_ranking_loss,
            pairwise_policy_ranking_weight=pairwise_policy_ranking_weight,
            pairwise_policy_margin=pairwise_policy_margin,
            pairwise_policy_min_target_gap=pairwise_policy_min_target_gap,
            oracle_best_action_loss=oracle_best_action_loss,
            oracle_best_action_loss_weight=oracle_best_action_loss_weight,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_hat.size(0)
        n += y_hat.size(0)

    return total_loss / max(1, n)


def eval_epoch(
    model,
    task,
    loader,
    device,
    max_neighbors: int,
    neighbor_sampling: bool,
    num_neighbors: List[int],
    seed_count: int,
    seed_strategy: str,
    max_sub_batches: int,
    seed_batch_size: int,
    task_name: str,
    aux_policy_loss: bool,
    aux_policy_loss_weight: float,
    aux_policy_target_name: str,
    pairwise_policy_ranking_loss: bool,
    pairwise_policy_ranking_weight: float,
    pairwise_policy_margin: float,
    pairwise_policy_min_target_gap: float,
    oracle_best_action_loss: bool,
    oracle_best_action_loss_weight: float,
    progress_prefix: str = None,
):
    model.eval()
    total_loss = 0.0
    n = 0

    collected_preds = []
    collected_labels: Dict[str, List[Any]] = {}
    collected_graphs = []

    num_batches = len(loader) if hasattr(loader, "__len__") else None
    if progress_prefix is not None and num_batches is not None:
        print(f"DT_PROGRESS_{progress_prefix}_META batches={num_batches}", flush=True)

    with torch.no_grad():
        for batch_idx, (batched_graphs, labels_dict) in enumerate(loader):
            if progress_prefix is not None:
                print(f"DT_PROGRESS_{progress_prefix}_BATCH {batch_idx + 1}", flush=True)

            if neighbor_sampling:
                H = build_day_embeddings_neighbor_sampling(
                    model=model,
                    batched_graphs=batched_graphs,
                    device=device,
                    num_neighbors=num_neighbors,
                    seed_count=seed_count,
                    seed_strategy=seed_strategy,
                    max_sub_batches=max_sub_batches,
                    seed_batch_size=seed_batch_size,
                )
                action_features = _extract_action_features(batched_graphs, labels_dict, device=device)
                state_summary_features = _extract_state_summary_features(batched_graphs, labels_dict, device=device)
                y_hat = model.forward_from_day_embeddings(
                    H,
                    action_features=action_features,
                    state_summary_features=state_summary_features,
                )
                batched_graphs_dev = [g.to(device) for g in batched_graphs]
            else:
                batched_graphs = subsample_neighbors_in_batches(batched_graphs, max_neighbors)
                batched_graphs_dev = [g.to(device) for g in batched_graphs]
                action_features = _extract_action_features(batched_graphs_dev, labels_dict, device=device)
                state_summary_features = _extract_state_summary_features(batched_graphs_dev, labels_dict, device=device)
                y_hat = model(
                    batched_graphs_dev,
                    action_features=action_features,
                    state_summary_features=state_summary_features,
                )

            labels_dict = _move_labels_dict_to_device(labels_dict, device)

            loss = _compute_total_loss(
                model=model,
                task=task,
                y_hat=y_hat,
                batched_graphs_dev=batched_graphs_dev,
                labels_dict=labels_dict,
                task_name=task_name,
                aux_policy_loss=aux_policy_loss,
                aux_policy_loss_weight=aux_policy_loss_weight,
                aux_policy_target_name=aux_policy_target_name,
                pairwise_policy_ranking_loss=pairwise_policy_ranking_loss,
                pairwise_policy_ranking_weight=pairwise_policy_ranking_weight,
                pairwise_policy_margin=pairwise_policy_margin,
                pairwise_policy_min_target_gap=pairwise_policy_min_target_gap,
                oracle_best_action_loss=oracle_best_action_loss,
                oracle_best_action_loss_weight=oracle_best_action_loss_weight,
            )

            total_loss += loss.item() * y_hat.size(0)
            n += y_hat.size(0)

            collected_preds.append(y_hat.cpu())
            collected_graphs.append([g.cpu() for g in batched_graphs_dev])

            for k, v in labels_dict.items():
                if torch.is_tensor(v):
                    collected_labels.setdefault(k, []).append(v.cpu())
                else:
                    collected_labels.setdefault(k, []).append(v)

    if len(collected_preds) == 0:
        empty_yhat = torch.empty((0, task.out_dim), dtype=torch.float32)
        return 0.0, {}, empty_yhat, [], {}

    y_hat_all = torch.cat(collected_preds, dim=0)
    labels_dict_all = _merge_collected_labels(collected_labels) if collected_labels else {}
    metrics = task.compute_eval_metrics(y_hat_all, collected_graphs, labels_dict_all)

    return total_loss / max(1, n), metrics, y_hat_all, collected_graphs, labels_dict_all

def infer_feature_dims(dataset):
    graphs, _ = dataset[0]
    g = graphs[0]
    in_channels = g.x.size(1)
    edge_dim = g.edge_attr.size(1) if hasattr(g, "edge_attr") else 0
    return in_channels, edge_dim


def infer_action_feature_dim(dataset) -> int:
    graphs, _ = dataset[0]
    g = graphs[0]
    feat = getattr(g, "action_features", None)
    if feat is None:
        return 0
    if torch.is_tensor(feat):
        if feat.dim() == 1:
            return int(feat.numel())
        return int(feat.size(-1))
    return 0


def infer_state_summary_feature_dim(dataset) -> int:
    graphs, _ = dataset[0]
    g = graphs[0]
    feat = getattr(g, "state_summary_features", None)
    if feat is None:
        return 0
    if torch.is_tensor(feat):
        if feat.dim() == 1:
            return int(feat.numel())
        return int(feat.size(-1))
    return 0


def _extract_action_features(
    batched_graphs: List,
    labels_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Prefer per-sample action features emitted by the dataset/collate path.
    Fall back to graph-attached action_features only when labels_dict does not carry them.
    """
    feat = labels_dict.get("action_features", None)
    if torch.is_tensor(feat):
        feat = feat.to(device=device, dtype=torch.float32)
        if feat.dim() == 1:
            feat = feat.view(1, -1)
        return feat

    if len(batched_graphs) == 0:
        return None

    g0 = batched_graphs[0]
    feat = getattr(g0, "action_features", None)
    if feat is None or not torch.is_tensor(feat):
        return None

    feat = feat.to(device=device, dtype=torch.float32)
    if feat.dim() == 1:
        feat = feat.view(1, -1)
    return feat


def _extract_state_summary_features(
    batched_graphs: List,
    labels_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Prefer per-sample explicit state-summary features emitted by the dataset/collate path.
    Fall back to graph-attached state_summary_features only when labels_dict does not carry them.
    """
    feat = labels_dict.get("state_summary_features", None)
    if torch.is_tensor(feat):
        feat = feat.to(device=device, dtype=torch.float32)
        if feat.dim() == 1:
            feat = feat.view(1, -1)
        return feat

    if len(batched_graphs) == 0:
        return None

    g0 = batched_graphs[0]
    feat = getattr(g0, "state_summary_features", None)
    if feat is None or not torch.is_tensor(feat):
        return None

    feat = feat.to(device=device, dtype=torch.float32)
    if feat.dim() == 1:
        feat = feat.view(1, -1)
    return feat


def _validate_task_labels(task: BaseTask, data_folder: str, dataset: Optional[TemporalGraphDataset] = None) -> None:
    sample_path: Optional[str] = None

    if dataset is not None and getattr(dataset, "use_policy_manifest", False):
        samples = getattr(dataset, "samples", [])
        if len(samples) == 0:
            raise FileNotFoundError(f"No policy-manifest samples found in data_folder: {data_folder}")

        first_sample = samples[0]
        file_keys = first_sample.get("file_keys", [])
        if len(file_keys) == 0:
            raise FileNotFoundError(f"First policy-manifest sample has no PT paths in data_folder: {data_folder}")
        sample_path = str(file_keys[0])

        try:
            sample_graphs, sample_labels = dataset[0]
        except Exception as e:
            raise ValueError(
                f"Could not load the first policy-manifest sample from '{data_folder}' for task-label validation. "
                f"Error: {e}"
            ) from e

        try:
            _ = task.get_targets(sample_graphs, sample_labels)
            return
        except Exception as e:
            label_keys = sorted([str(k) for k in sample_labels.keys()]) if isinstance(sample_labels, dict) else []
            msg = (
                f"Selected task '{task.name}' requires a label that is missing in the policy-manifest sample "
                f"loaded from '{sample_path}'.\n"
                f"Error: {e}\n"
                f"Available manifest label keys include (truncated): {label_keys[:80]}"
            )
            raise ValueError(msg) from e

    try:
        pt_files = [f for f in os.listdir(data_folder) if f.endswith(".pt")]
    except Exception:
        pt_files = []
    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt files found in data_folder: {data_folder}")
    sample_path = os.path.join(data_folder, sorted(pt_files)[0])

    sample = torch.load(sample_path, weights_only=False)

    try:
        _ = task.get_targets([sample], {})
    except Exception as e:
        avail = sorted([a for a in dir(sample) if a.startswith("y_")])
        msg = (
            f"Selected task '{task.name}' requires a label that is missing in '{sample_path}'.\n"
            f"Error: {e}\n"
            f"Available y_* attributes include (truncated): {avail[:40]}"
        )
        raise ValueError(msg) from e




def _is_grouped_policy_classification_task(task_name: str) -> bool:
    task_norm = str(task_name).strip().lower()
    return task_norm.startswith("oracle_best_action_h")

def _auto_aux_policy_target_name(task_name: str) -> str:
    task_norm = str(task_name).strip().lower()
    m = re.search(r"_h(\d+)$", task_norm)
    h = int(m.group(1)) if m else 7

    if task_norm.startswith("endogenous_importation_majority_h") or task_norm.startswith("endogenous_importation_share_h"):
        return f"y_h{h}_endog_share"
    if task_norm.startswith("endogenous_transmission_majority_h") or task_norm.startswith("endogenous_transmission_share_h"):
        return f"y_h{h}_trans_share"
    if task_norm.startswith("mechanism_import_share_h"):
        return f"y_h{h}_import_share"
    if task_norm.startswith("mechanism_selection_share_h"):
        return f"y_h{h}_select_share"
    if task_norm.startswith("amr_cr_acq_h"):
        return f"y_h{h}_cr_acq"
    if task_norm.startswith("amr_ir_inf_h"):
        return f"y_h{h}_ir_inf"
    if task_norm.startswith("predict_resistance_emergence_h"):
        return f"y_h{h}_any_res_emergence"
    if task_norm.startswith("transmission_importation_resistant_burden_gain_h"):
        return f"y_h{h}_trans_import_res_gain"
    if task_norm.startswith("oracle_best_action_h"):
        return f"y_h{h}_trans_import_res_gain"
    if task_norm.startswith("transmission_importation_resistant_burden_h"):
        return f"y_h{h}_trans_import_res"
    if task_norm.startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac"
    if task_norm.startswith("optimal_screening_target_h"):
        return f"y_h{h}_screening_gain"
    if task_norm.startswith("staff_mediation_effect_h"):
        return f"y_h{h}_transmissions"
    return ""


def _task_primary_manifest_target_name(task_name: str) -> str:
    task_norm = str(task_name).strip().lower()
    m = re.search(r"_h(\d+)$", task_norm)
    h = int(m.group(1)) if m else 7

    if task_norm.startswith("transmission_importation_resistant_burden_gain_h"):
        return f"y_h{h}_trans_import_res_gain"
    if task_norm.startswith("oracle_best_action_h"):
        return f"oracle_best_action_index_h{h}"
    if task_norm.startswith("transmission_importation_resistant_burden_h"):
        return f"y_h{h}_trans_import_res"
    if task_norm.startswith("transmission_resistant_burden_gain_h"):
        return f"y_h{h}_trans_res_gain"
    if task_norm.startswith("transmission_resistant_burden_h"):
        return f"y_h{h}_trans_res"
    if task_norm.startswith("endogenous_importation_majority_h") or task_norm.startswith("endogenous_importation_share_h"):
        return f"y_h{h}_endog_share"
    if task_norm.startswith("endogenous_transmission_majority_h") or task_norm.startswith("endogenous_transmission_share_h"):
        return f"y_h{h}_trans_share"
    if task_norm.startswith("mechanism_import_share_h"):
        return f"y_h{h}_import_share"
    if task_norm.startswith("mechanism_selection_share_h"):
        return f"y_h{h}_select_share"
    if task_norm.startswith("amr_cr_acq_h"):
        return f"y_h{h}_cr_acq"
    if task_norm.startswith("amr_ir_inf_h"):
        return f"y_h{h}_ir_inf"
    if task_norm.startswith("predict_resistance_emergence_h"):
        return f"y_h{h}_any_res_emergence"
    if task_norm.startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac"
    if task_norm.startswith("optimal_screening_target_h"):
        return f"y_h{h}_screening_gain"
    if task_norm.startswith("staff_mediation_effect_h"):
        return f"y_h{h}_transmissions"
    return ""


def _should_skip_duplicate_aux_policy_loss(task_name: str, aux_target_name: str, task) -> bool:
    if bool(getattr(task, "is_classification", False)):
        return False
    aux_name = str(aux_target_name).strip().lower()
    primary_name = _task_primary_manifest_target_name(task_name)
    return aux_name != "" and aux_name == primary_name


def _auto_aux_policy_target_direction(task_name: str, aux_target_name: str) -> str:
    task_norm = str(task_name).strip().lower()
    name = str(aux_target_name).strip().lower()

    if name.endswith("_gain"):
        return "maximize"

    if "burden_gain" in task_norm:
        return "maximize"

    return "minimize"


def _policy_utility_from_logits(
    y_hat: torch.Tensor,
    task_name: str,
    aux_target_name: str,
    output_kind: str = "probability",
) -> torch.Tensor:
    """
    Return a policy-selection utility derived from the model output.

    Classification:
      - output_kind="probability": probability-like utility in [0, 1]
      - output_kind="score": unbounded margin utility for ranking

    Regression:
      - output_kind is ignored semantically; both variants return a monotone
        utility score derived from the scalar prediction.
      - For minimize-style oracle targets (e.g. burden), lower prediction means
        higher utility, so utility = -prediction.
      - For maximize-style targets (e.g. gain), utility = prediction.
    """
    direction = _auto_aux_policy_target_direction(task_name, aux_target_name)
    output_kind = str(output_kind).strip().lower()

    if y_hat.dim() == 1:
        pred_scalar = y_hat.view(-1)
        utility_score = pred_scalar if direction == "maximize" else -pred_scalar
        return utility_score

    if y_hat.dim() != 2:
        raise ValueError(f"Expected y_hat to have shape [N, C] or [N], got {tuple(y_hat.shape)}")

    if y_hat.size(1) == 1:
        pred_scalar = y_hat.view(-1)
        utility_score = pred_scalar if direction == "maximize" else -pred_scalar
        return utility_score

    probs = F.softmax(y_hat, dim=1)
    p1 = probs[:, 1]
    logit_margin = y_hat[:, 1] - y_hat[:, 0]

    if direction == "maximize":
        utility_prob = p1
        utility_score = logit_margin
    else:
        utility_prob = 1.0 - p1
        utility_score = -logit_margin

    if output_kind == "score":
        return utility_score
    return utility_prob


def _extract_policy_state_keys(batched_graphs, labels_dict) -> List[str]:
    state_ids = labels_dict.get("state_id", None)
    if torch.is_tensor(state_ids):
        keys = [str(x.item()) for x in state_ids.view(-1)]
    elif isinstance(state_ids, (list, tuple)):
        keys = [str(x) for x in state_ids]
    else:
        if len(batched_graphs) == 0:
            raise ValueError("Policy ranking loss requires state_id, but batched_graphs is empty.")
        data_list = batched_graphs[0].to_data_list()
        keys = []
        for g in data_list:
            sid = getattr(g, "state_id", None)
            keys.append("" if sid is None else str(sid))

    if len(keys) == 0 or any(str(k).strip() == "" for k in keys):
        raise ValueError(
            "Policy ranking loss requires a valid state_id for every row in the batch. "
            "At least one state_id is missing or empty."
        )
    return keys


def _labels_dict_to_long_tensor(labels_dict, key: str, device: torch.device) -> Optional[torch.Tensor]:
    raw = labels_dict.get(key, None)
    if raw is None:
        return None
    if torch.is_tensor(raw):
        return raw.to(device=device, dtype=torch.long).view(-1)
    if isinstance(raw, (list, tuple)):
        vals: List[int] = []
        for x in raw:
            if torch.is_tensor(x):
                vals.append(int(x.view(-1)[0].item()))
            else:
                vals.append(int(x))
        return torch.tensor(vals, device=device, dtype=torch.long)
    raise ValueError(f"Expected labels_dict['{key}'] to be a tensor/list/tuple, got {type(raw)}")


def _extract_policy_action_index_tensor(labels_dict, device: torch.device, n_rows: int) -> torch.Tensor:
    action_index = _labels_dict_to_long_tensor(labels_dict, "action_index", device)
    if action_index is None:
        raise ValueError("Policy ranking loss requires action_index in labels_dict, but it is missing.")
    if action_index.numel() != n_rows:
        raise ValueError(
            f"Policy ranking loss size mismatch: action_index has {action_index.numel()} rows, "
            f"but model output has {n_rows} rows."
        )
    return action_index


def _extract_grouped_oracle_action_index_tensor(
    labels_dict,
    task_name: str,
    device: torch.device,
    n_rows: int,
) -> Optional[torch.Tensor]:
    target_name = _task_primary_manifest_target_name(task_name)
    if not str(target_name).startswith("oracle_best_action_index_h"):
        return None
    oracle_action_index = _labels_dict_to_long_tensor(labels_dict, target_name, device)
    if oracle_action_index is None:
        raise ValueError(
            f"Grouped oracle-best-action supervision requires '{target_name}' in labels_dict, but it is missing."
        )
    if oracle_action_index.numel() != n_rows:
        raise ValueError(
            f"Grouped oracle-best-action size mismatch: '{target_name}' has {oracle_action_index.numel()} rows, "
            f"but model output has {n_rows} rows."
        )
    return oracle_action_index


def _resolve_group_target_local_position(
    group_action_idx: torch.Tensor,
    group_target_idx: torch.Tensor,
    context: str,
) -> int:
    if group_action_idx.numel() == 0 or group_target_idx.numel() == 0:
        raise ValueError(f"{context} received an empty grouped action set.")

    oracle_class = int(group_target_idx[0].item())
    if not bool(torch.all(group_target_idx == oracle_class).item()):
        raise ValueError(f"{context} found inconsistent oracle action indices within a single state batch.")

    matches = (group_action_idx == oracle_class).nonzero(as_tuple=False).view(-1)
    if int(matches.numel()) != 1:
        raise ValueError(
            f"{context} expected exactly one action_index={oracle_class} inside a state batch, "
            f"but found {int(matches.numel())}."
        )
    return int(matches[0].item())


class _PolicyStateBatchSampler(Sampler[List[int]]):
    def __init__(self, batches: List[List[int]], shuffle_batches: bool = True):
        self.batches = [list(b) for b in batches if len(b) > 0]
        self.shuffle_batches = bool(shuffle_batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle_batches:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]

    def __len__(self) -> int:
        return len(self.batches)


def _build_policy_state_batches_from_subset(subset: Subset) -> List[List[int]]:
    """
    Build one batch per complete policy state.
    Each batch contains all rows for the same state_id.
    This ensures the ranking loss always sees the full action set for a state.
    """
    dataset = subset.dataset
    if not getattr(dataset, "use_policy_manifest", False):
        raise ValueError("_build_policy_state_batches_from_subset requires a policy-manifest dataset.")

    groups: Dict[str, List[int]] = defaultdict(list)

    for local_subset_idx, dataset_idx in enumerate(subset.indices):
        sample = dataset.samples[int(dataset_idx)]
        row_meta = sample.get("row_meta", {})
        state_id = str(row_meta.get("state_id", "")).strip()
        if state_id == "":
            raise ValueError(f"Missing state_id in policy-manifest row for dataset index {dataset_idx}.")
        groups[state_id].append(local_subset_idx)

    batches: List[List[int]] = []
    for state_id in sorted(groups.keys()):
        batch = sorted(groups[state_id])
        batches.append(batch)

    return batches


def _expected_policy_action_ids(dataset: TemporalGraphDataset) -> List[str]:
    """
    Derive the global expected candidate action menu from the full policy manifest.
    """
    if not getattr(dataset, "use_policy_manifest", False):
        return []

    action_ids = sorted({
        str(sample.get("row_meta", {}).get("action_id", "")).strip()
        for sample in getattr(dataset, "samples", [])
        if str(sample.get("row_meta", {}).get("action_id", "")).strip() != ""
    })
    return action_ids


def _filter_policy_manifest_indices_by_complete_action_set(
    dataset: TemporalGraphDataset,
    indices: List[int],
    split_name: str,
    require_complete_action_set: bool,
) -> Tuple[List[int], List[Dict[str, Any]], List[str]]:
    """
    Keep only states whose action rows exactly match the global expected action menu.

    This prevents incomplete or duplicate action sets from entering training,
    validation, or test evaluation, which would otherwise corrupt within-state
    oracle/ranking supervision.
    """
    if not getattr(dataset, "use_policy_manifest", False):
        return sorted(indices), [], []

    expected_action_ids = _expected_policy_action_ids(dataset)
    expected_action_set = set(expected_action_ids)

    groups: Dict[str, List[int]] = defaultdict(list)
    for dataset_idx in indices:
        sample = dataset.samples[int(dataset_idx)]
        row_meta = sample.get("row_meta", {})
        state_id = str(row_meta.get("state_id", "")).strip()
        if state_id == "":
            raise ValueError(
                f"Missing state_id in policy-manifest row for dataset index {dataset_idx} "
                f"while filtering split='{split_name}'."
            )
        groups[state_id].append(int(dataset_idx))

    kept: List[int] = []
    dropped: List[Dict[str, Any]] = []

    for state_id in sorted(groups.keys()):
        state_indices = sorted(groups[state_id])
        state_action_ids: List[str] = []
        for dataset_idx in state_indices:
            row_meta = dataset.samples[int(dataset_idx)].get("row_meta", {})
            action_id = str(row_meta.get("action_id", "")).strip()
            if action_id == "":
                action_id = "<missing>"
            state_action_ids.append(action_id)

        unique_action_ids = sorted(set(state_action_ids))
        has_duplicates = len(unique_action_ids) != len(state_action_ids)
        exact_match = set(unique_action_ids) == expected_action_set and (not has_duplicates)

        if require_complete_action_set and (not exact_match):
            missing = sorted(expected_action_set - set(unique_action_ids))
            extra = sorted(set(unique_action_ids) - expected_action_set)
            dropped.append(
                {
                    "split": str(split_name),
                    "state_id": str(state_id),
                    "n_rows": int(len(state_indices)),
                    "n_unique_actions": int(len(unique_action_ids)),
                    "has_duplicates": bool(has_duplicates),
                    "expected_n_actions": int(len(expected_action_ids)),
                    "expected_action_ids": list(expected_action_ids),
                    "observed_action_ids": list(unique_action_ids),
                    "missing_action_ids": list(missing),
                    "extra_action_ids": list(extra),
                }
            )
            continue

        kept.extend(state_indices)

    return sorted(kept), dropped, expected_action_ids


def _make_train_loader(
    dataset: TemporalGraphDataset,
    train_set: Subset,
    batch_size: int,
    expected_action_ids: Optional[List[str]] = None,
):
    if getattr(dataset, "use_policy_manifest", False):
        if expected_action_ids is None:
            expected_action_ids = _expected_policy_action_ids(dataset)
        train_state_batches = _build_policy_state_batches_from_subset(train_set)
        train_batch_sampler = _PolicyStateBatchSampler(
            batches=train_state_batches,
            shuffle_batches=True,
        )
        return DataLoader(
            train_set,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_temporal_graph_batch,
        )

    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_temporal_graph_batch,
    )


def _make_eval_loader(
    subset_or_dataset,
    batch_size: int,
    expected_action_ids: Optional[List[str]] = None,
):
    dataset = getattr(subset_or_dataset, "dataset", subset_or_dataset)
    if getattr(dataset, "use_policy_manifest", False) and isinstance(subset_or_dataset, Subset):
        if expected_action_ids is None:
            expected_action_ids = _expected_policy_action_ids(dataset)
        eval_state_batches = _build_policy_state_batches_from_subset(subset_or_dataset)
        eval_batch_sampler = _PolicyStateBatchSampler(
            batches=eval_state_batches,
            shuffle_batches=False,
        )
        return DataLoader(
            subset_or_dataset,
            batch_sampler=eval_batch_sampler,
            collate_fn=collate_temporal_graph_batch,
        )

    return DataLoader(
        subset_or_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_temporal_graph_batch,
    )


def _zscore_policy_target_within_state(
    *,
    target: torch.Tensor,
    batched_graphs,
    labels_dict,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Z-score the policy target within each decision state.

    This preserves within-state action ordering while removing state-level
    offset and scale. States with near-zero variance are mean-centered only.
    """
    target = target.to(dtype=torch.float32).view(-1)
    state_keys = _extract_policy_state_keys(batched_graphs=batched_graphs, labels_dict=labels_dict)

    if len(state_keys) != target.numel():
        raise ValueError(
            f"Within-state z-score size mismatch: extracted {len(state_keys)} state_ids, "
            f"but target has {target.numel()} rows."
        )

    out = target.clone()
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, key in enumerate(state_keys):
        groups[str(key)].append(idx)

    for idxs in groups.values():
        if len(idxs) == 0:
            continue

        idx_t = torch.as_tensor(idxs, device=target.device, dtype=torch.long)
        vals = target.index_select(0, idx_t)

        finite_mask = torch.isfinite(vals)
        if int(finite_mask.sum().item()) == 0:
            continue

        valid_vals = vals[finite_mask]
        mu = valid_vals.mean()

        if valid_vals.numel() < 2:
            vals_z = vals - mu
        else:
            sigma = valid_vals.std(unbiased=False)
            if float(sigma.item()) <= eps:
                vals_z = vals - mu
            else:
                vals_z = (vals - mu) / (sigma + eps)

        out.index_copy_(0, idx_t, vals_z)

    return out

def _compute_pairwise_policy_ranking_loss(
    *,
    y_hat: torch.Tensor,
    batched_graphs,
    labels_dict,
    task_name: str,
    aux_target_name: str,
    pairwise_policy_margin: float,
    pairwise_policy_min_target_gap: float,
) -> torch.Tensor:
    aux_target = labels_dict.get(aux_target_name, None)
    if aux_target is None:
        raise ValueError(
            f"Policy ranking loss requires aux target '{aux_target_name}', but it is missing from labels_dict."
        )
    if not torch.is_tensor(aux_target):
        raise ValueError(
            f"Policy ranking loss requires aux target '{aux_target_name}' to be a tensor, "
            f"got {type(aux_target)} instead."
        )

    aux_target = aux_target.to(device=y_hat.device, dtype=torch.float32).view(-1)
    if aux_target.numel() != y_hat.size(0):
        raise ValueError(
            f"Policy ranking loss size mismatch: aux target '{aux_target_name}' has "
            f"{aux_target.numel()} rows, but model output has {y_hat.size(0)} rows."
        )

    state_keys = _extract_policy_state_keys(batched_graphs=batched_graphs, labels_dict=labels_dict)
    if len(state_keys) != y_hat.size(0):
        raise ValueError(
            f"Policy ranking loss size mismatch: extracted {len(state_keys)} state_ids, "
            f"but model output has {y_hat.size(0)} rows."
        )

    utility_score = _policy_utility_from_logits(
        y_hat=y_hat,
        task_name=task_name,
        aux_target_name=aux_target_name,
        output_kind="score",
    ).view(-1)

    direction = _auto_aux_policy_target_direction(task_name, aux_target_name)
    target_gap_min = float(max(0.0, pairwise_policy_min_target_gap))
    margin = float(max(0.0, pairwise_policy_margin))
    grouped_policy_task = _is_grouped_policy_classification_task(task_name)
    action_index = _extract_policy_action_index_tensor(labels_dict, y_hat.device, y_hat.size(0))
    oracle_action_index = _extract_grouped_oracle_action_index_tensor(
        labels_dict,
        task_name=task_name,
        device=y_hat.device,
        n_rows=y_hat.size(0),
    )

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, key in enumerate(state_keys):
        groups[str(key)].append(idx)

    pair_losses: List[torch.Tensor] = []
    listwise_losses: List[torch.Tensor] = []

    for idxs in groups.values():
        if len(idxs) < 2:
            continue
    
        raw_action_ids = labels_dict.get("action_id", None)
        if raw_action_ids is None:
            raise ValueError("Policy ranking loss requires action_id in labels_dict, but it is missing.")

        if torch.is_tensor(raw_action_ids):
            sel = torch.as_tensor(idxs, device=raw_action_ids.device, dtype=torch.long)
            state_action_ids = [str(x.item()) for x in raw_action_ids.view(-1).index_select(0, sel).cpu().tolist()]
        elif isinstance(raw_action_ids, (list, tuple)):
            state_action_ids = [str(raw_action_ids[i]) for i in idxs]
        else:
            raise ValueError("Policy ranking loss requires action_id as a tensor/list/tuple in labels_dict.")

        if any(str(a).strip() == "" for a in state_action_ids):
            raise ValueError("Policy ranking loss found an empty action_id inside a state batch.")

        if len(set(state_action_ids)) != len(state_action_ids):
            raise ValueError(
                f"Policy ranking loss expected one row per action within a state batch, "
                f"but found duplicates: {state_action_ids}"
            )

        idx_t = torch.as_tensor(idxs, device=y_hat.device, dtype=torch.long)
        tvals = aux_target.index_select(0, idx_t)
        svals = utility_score.index_select(0, idx_t)
        group_action_idx = action_index.index_select(0, idx_t)
        group_oracle_idx = None if oracle_action_index is None else oracle_action_index.index_select(0, idx_t)

        finite_mask = torch.isfinite(tvals) & torch.isfinite(svals)
        if int(finite_mask.sum().item()) < 2:
            continue

        tvals = tvals[finite_mask]
        svals = svals[finite_mask]
        group_action_idx = group_action_idx[finite_mask]
        if group_oracle_idx is not None:
            group_oracle_idx = group_oracle_idx[finite_mask]

        diff_target = tvals.unsqueeze(1) - tvals.unsqueeze(0)
        if direction == "minimize":
            better_mask = diff_target < -target_gap_min
            best_local = int(torch.argmin(tvals).item())
        else:
            better_mask = diff_target > target_gap_min
            best_local = int(torch.argmax(tvals).item())

        if torch.any(better_mask):
            diff_score = svals.unsqueeze(1) - svals.unsqueeze(0)
            pair_loss = F.relu(margin - diff_score)
            pair_losses.append(pair_loss[better_mask].mean())

        if grouped_policy_task and group_oracle_idx is not None:
            best_local = _resolve_group_target_local_position(
                group_action_idx=group_action_idx,
                group_target_idx=group_oracle_idx,
                context="Policy ranking loss",
            )

        logits = svals.view(1, -1)
        target_idx = torch.tensor([best_local], device=y_hat.device, dtype=torch.long)
        listwise_losses.append(F.cross_entropy(logits, target_idx))

    components: List[torch.Tensor] = []
    if len(pair_losses) > 0:
        components.append(torch.stack(pair_losses, dim=0).mean())
    if len(listwise_losses) > 0:
        components.append(torch.stack(listwise_losses, dim=0).mean())

    if len(components) == 0:
        return y_hat.new_zeros(())
    return torch.stack(components, dim=0).mean()


def _compute_oracle_best_action_membership_loss(
    *,
    y_hat: torch.Tensor,
    batched_graphs,
    labels_dict,
    task_name: str,
    aux_target_name: str,
) -> torch.Tensor:
    """
    True within-state listwise supervision.

    For each policy state, the model produces one utility score per candidate
    action. This loss applies a softmax over that action set and optimizes a
    cross-entropy objective against the oracle-best action index for that state.

    This directly matches downstream policy selection:
      choose the best action within each state by comparing all candidates
      together, rather than learning rowwise one-vs-rest membership targets.
    """
    aux_target = labels_dict.get(aux_target_name, None)
    if aux_target is None:
        raise ValueError(
            f"Oracle-best-action loss requires aux target '{aux_target_name}', but it is missing from labels_dict."
        )
    if not torch.is_tensor(aux_target):
        raise ValueError(
            f"Oracle-best-action loss requires aux target '{aux_target_name}' to be a tensor, "
            f"got {type(aux_target)} instead."
        )

    aux_target = aux_target.to(device=y_hat.device, dtype=torch.float32).view(-1)
    if aux_target.numel() != y_hat.size(0):
        raise ValueError(
            f"Oracle-best-action loss size mismatch: aux target '{aux_target_name}' has "
            f"{aux_target.numel()} rows, but model output has {y_hat.size(0)} rows."
        )

    state_keys = _extract_policy_state_keys(batched_graphs=batched_graphs, labels_dict=labels_dict)
    if len(state_keys) != y_hat.size(0):
        raise ValueError(
            f"Oracle-best-action loss size mismatch: extracted {len(state_keys)} state_ids, "
            f"but model output has {y_hat.size(0)} rows."
        )

    utility_score = _policy_utility_from_logits(
        y_hat=y_hat,
        task_name=task_name,
        aux_target_name=aux_target_name,
        output_kind="score",
    ).view(-1)

    action_index = _extract_policy_action_index_tensor(labels_dict, y_hat.device, y_hat.size(0))
    oracle_action_index = _extract_grouped_oracle_action_index_tensor(
        labels_dict,
        task_name=task_name,
        device=y_hat.device,
        n_rows=y_hat.size(0),
    )
    if oracle_action_index is None:
        raise ValueError(
            f"Oracle-best-action loss requires grouped oracle labels for task '{task_name}', but none were found."
        )

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, key in enumerate(state_keys):
        groups[str(key)].append(idx)

    losses: List[torch.Tensor] = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue

        idx_t = torch.as_tensor(idxs, device=y_hat.device, dtype=torch.long)
        tvals = aux_target.index_select(0, idx_t)
        svals = utility_score.index_select(0, idx_t)
        group_action_idx = action_index.index_select(0, idx_t)
        group_oracle_idx = oracle_action_index.index_select(0, idx_t)

        finite_mask = torch.isfinite(tvals) & torch.isfinite(svals)
        if int(finite_mask.sum().item()) < 2:
            continue

        svals = svals[finite_mask]
        group_action_idx = group_action_idx[finite_mask]
        group_oracle_idx = group_oracle_idx[finite_mask]

        best_local = _resolve_group_target_local_position(
            group_action_idx=group_action_idx,
            group_target_idx=group_oracle_idx,
            context="Oracle-best-action loss",
        )

        logits = svals.view(1, -1)
        target_idx = torch.tensor([best_local], device=y_hat.device, dtype=torch.long)
        losses.append(F.cross_entropy(logits, target_idx))

    if len(losses) == 0:
        return y_hat.new_zeros(())
    return torch.stack(losses, dim=0).mean()


def _compute_total_loss(
    *,
    model,
    task,
    y_hat: torch.Tensor,
    batched_graphs_dev,
    labels_dict,
    task_name: str,
    aux_policy_loss: bool,
    aux_policy_loss_weight: float,
    aux_policy_target_name: str,
    pairwise_policy_ranking_loss: bool,
    pairwise_policy_ranking_weight: float,
    pairwise_policy_margin: float,
    pairwise_policy_min_target_gap: float,
    oracle_best_action_loss: bool,
    oracle_best_action_loss_weight: float,
) -> torch.Tensor:
    grouped_policy_task = _is_grouped_policy_classification_task(task_name)
    if grouped_policy_task:
        if not bool(getattr(model, "use_action_conditioning", False)):
            raise ValueError("Grouped oracle-best-action training requires --use_action_conditioning true.")
        base_loss = y_hat.new_zeros(())
    else:
        base_loss = task.compute_loss(y_hat, batched_graphs_dev, labels_dict)

    if not bool(getattr(model, "use_action_conditioning", False)):
        return base_loss

    aux_name = str(aux_policy_target_name).strip() or _auto_aux_policy_target_name(task_name)
    if aux_name == "":
        return base_loss

    if grouped_policy_task and not bool(oracle_best_action_loss):
        raise ValueError(
            f"Task '{task_name}' is a grouped oracle-best-action classification task and requires --oracle_best_action_loss true."
        )

    policy_primary_mode = bool(oracle_best_action_loss) or grouped_policy_task
    total_loss = y_hat.new_zeros(()) if policy_primary_mode else base_loss

    skip_duplicate_aux_loss = _should_skip_duplicate_aux_policy_loss(
        task_name=task_name,
        aux_target_name=aux_name,
        task=task,
    )

    aux_target = labels_dict.get(aux_name, None)
    aux_target_z = None
    if aux_target is not None and torch.is_tensor(aux_target):
        aux_target = aux_target.to(device=y_hat.device, dtype=torch.float32).view(-1)
        aux_target_z = _zscore_policy_target_within_state(
            target=aux_target,
            batched_graphs=batched_graphs_dev,
            labels_dict=labels_dict,
        )

    if aux_policy_loss and (not skip_duplicate_aux_loss) and aux_target_z is not None:
        if y_hat.dim() == 1:
            pred_scalar = y_hat.view(-1)
        elif y_hat.dim() == 2 and y_hat.size(0) == aux_target_z.size(0) and y_hat.size(1) == 1:
            pred_scalar = y_hat.view(-1)
        else:
            pred_scalar = None

        if task.is_classification:
            if y_hat.dim() == 2 and y_hat.size(0) == aux_target_z.size(0):
                utility_prob = _policy_utility_from_logits(
                    y_hat=y_hat,
                    task_name=task_name,
                    aux_target_name=aux_name,
                    output_kind="probability",
                ).view(-1)

                aux_target_match = torch.sigmoid(aux_target_z)
                aux_loss = F.mse_loss(utility_prob, aux_target_match)
                total_loss = total_loss + float(aux_policy_loss_weight) * aux_loss
        else:
            if pred_scalar is not None and pred_scalar.numel() == aux_target_z.numel():
                pred_mu = pred_scalar.mean()
                pred_sigma = pred_scalar.std(unbiased=False)
                if float(pred_sigma.item()) <= 1e-6:
                    pred_scalar_z = pred_scalar - pred_mu
                else:
                    pred_scalar_z = (pred_scalar - pred_mu) / (pred_sigma + 1e-6)

                aux_loss = F.mse_loss(pred_scalar_z, aux_target_z)
                total_loss = total_loss + float(aux_policy_loss_weight) * aux_loss

    policy_labels_dict = labels_dict
    if aux_target_z is not None:
        policy_labels_dict = dict(labels_dict)
        policy_labels_dict[aux_name] = aux_target_z

    if oracle_best_action_loss:
        oracle_best_loss = _compute_oracle_best_action_membership_loss(
            y_hat=y_hat,
            batched_graphs=batched_graphs_dev,
            labels_dict=policy_labels_dict,
            task_name=task_name,
            aux_target_name=aux_name,
        )
        total_loss = total_loss + float(oracle_best_action_loss_weight) * oracle_best_loss

    if pairwise_policy_ranking_loss:
        ranking_loss = _compute_pairwise_policy_ranking_loss(
            y_hat=y_hat,
            batched_graphs=batched_graphs_dev,
            labels_dict=policy_labels_dict,
            task_name=task_name,
            aux_target_name=aux_name,
            pairwise_policy_margin=pairwise_policy_margin,
            pairwise_policy_min_target_gap=pairwise_policy_min_target_gap,
        )
        total_loss = total_loss + float(pairwise_policy_ranking_weight) * ranking_loss

    return total_loss


def _get_window_sim_id(dataset: TemporalGraphDataset, window_fnames: List[str], cache: Dict[str, str]) -> str:
    """
    Determine the trajectory id for a window.
    Prefer reading Data.sim_id from the first file in the window.
    """
    if not window_fnames:
        return "unknown"
    f0 = window_fnames[0]
    if f0 in cache:
        return cache[f0]

    path = dataset._disk_paths.get(f0, None)
    if path is None:
        cache[f0] = "unknown"
        return "unknown"

    try:
        d = torch.load(path, weights_only=False)
    except Exception:
        cache[f0] = "unknown"
        return "unknown"

    sim_id = getattr(d, "sim_id", None)
    if sim_id is not None and str(sim_id).strip() != "":
        cache[f0] = str(sim_id)
        return cache[f0]

    m = re.match(r"(.+?)_t(\d+)", str(f0))
    prefix = f0 if m is None else m.group(1)
    cache[f0] = str(prefix)
    return cache[f0]




def _metadata_list_from_graph_attr(g_data, attr_name: str, expected_len: int) -> List[Any]:
    """
    Recover node-aligned Python-list metadata from an individual PyG Data object.

    Handles:
      - already-flat Python lists/tuples
      - tensors (converted to Python lists)
      - missing attributes (returns empty list)
    """
    raw = getattr(g_data, attr_name, None)
    if raw is None:
        return []
    if torch.is_tensor(raw):
        try:
            out = raw.detach().cpu().tolist()
        except Exception:
            return []
        return list(out) if isinstance(out, list) else [out]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        return list(raw)
    except Exception:
        return []


def _infer_staff_role_from_name(node_name: str) -> str:
    """
    Fallback role inference using simulator naming conventions when explicit role
    metadata is absent or malformed.
    """
    s = str(node_name).strip().lower()
    if s.startswith("s"):
        return "staff"
    return "patient"


def _is_staff_role(role_value: Any, node_name: Optional[str] = None) -> bool:
    s = str(role_value).strip().lower()
    if s in {"staff", "hcw", "healthcare_worker", "health-care-worker", "worker"}:
        return True
    if "staff" in s:
        return True
    if node_name is not None:
        return _infer_staff_role_from_name(str(node_name)) == "staff"
    return False


def _normalize_role_value(role_value: Any, node_name: Optional[str] = None) -> str:
    if _is_staff_role(role_value, node_name=node_name):
        return "staff"
    s = str(role_value).strip().lower()
    if s == "patient":
        return "patient"
    if node_name is not None:
        return _infer_staff_role_from_name(str(node_name))
    return "unknown"


def _infer_state_signal_from_features(x_cpu: torch.Tensor) -> Tuple[List[float], str]:
    if x_cpu is None or x_cpu.ndim != 2 or x_cpu.size(0) == 0:
        return [], "unknown"

    x_cpu = x_cpu.to(torch.float32)

    # New ground-truth layout from patched convert_to_pt.py:
    # [is_staff, amr_U, amr_CS, amr_CR, amr_IS, amr_IR, abx, iso, new_cr, new_ir, ward_id_norm, ward_cover_norm]
    if x_cpu.size(1) >= 12:
        cr_positive = x_cpu[:, 3]
        return [float(v) for v in cr_positive.detach().cpu().tolist()], "cr_positive"

    # New partial-observation layout:
    # [is_staff, obs_positive, abx, iso, new_cr, new_ir, ward_id_norm, ward_cover_norm]
    if x_cpu.size(1) >= 8:
        obs_positive = x_cpu[:, 1]
        return [float(v) for v in obs_positive.detach().cpu().tolist()], "observed_positive"

    # Legacy fallback
    state_col = x_cpu[:, 1] if x_cpu.size(1) >= 2 else torch.zeros((x_cpu.size(0),), dtype=torch.float32)
    state_vals = [float(v) for v in state_col.detach().cpu().tolist()]
    if len(state_vals) == 0:
        return [], "unknown"

    rounded_unique = sorted({int(round(v)) for v in state_vals})
    if all(v in {0, 1, 2, 3, 4} for v in rounded_unique) and any(v >= 2 for v in rounded_unique):
        state_signal = [1.0 if int(round(v)) == 2 else 0.0 for v in state_vals]
        return state_signal, "cr_positive"

    if all(v in {0, 1} for v in rounded_unique):
        state_signal = [1.0 if int(round(v)) == 1 else 0.0 for v in state_vals]
        return state_signal, "observed_positive"

    if max(state_vals) > 1.0:
        state_signal = [1.0 if int(round(v)) == 2 else 0.0 for v in state_vals]
        return state_signal, "cr_positive"

    return [float(v) for v in state_vals], "observed_positive"


def _run_fullgraph_day_embeddings_and_attention(model, batched_graphs, labels_dict, device):
    model.eval()
    day_embs = []
    day_attn = []
    day_batches = []
    day_node_names = []
    day_node_roles = []
    day_ward_id = []
    day_ward_ids = []
    day_ward_cover_count = []
    day_node_state_signal = []
    day_node_state_signal_name = []

    for g in batched_graphs:
        gd = g.to(device)
        pooled, attn = model.gnn(
            gd.x,
            gd.edge_index,
            getattr(gd, 'edge_attr', None),
            gd.batch,
            return_attention=True,
        )
        day_embs.append(pooled)
        day_attn.append(attn.detach().cpu())
        day_batches.append(gd.batch.detach().cpu())

        num_graphs = int(g.num_graphs) if hasattr(g, "num_graphs") else int(g.batch.max().item()) + 1
        names_per_graph: List[List[str]] = []
        roles_per_graph: List[List[str]] = []
        ward_id_per_graph: List[List[int]] = []
        ward_ids_per_graph: List[List[str]] = []
        ward_cover_per_graph: List[List[int]] = []
        state_signal_per_graph: List[List[float]] = []
        state_signal_name_per_graph: List[str] = []

        data_list = g.to_data_list()
        if len(data_list) != num_graphs:
            raise RuntimeError(
                f"Metadata reconstruction mismatch: expected {num_graphs} graphs in batch, got {len(data_list)}"
            )

        for g_data in data_list:
            n_nodes = int(g_data.num_nodes)

            names_raw = _metadata_list_from_graph_attr(g_data, "node_names", n_nodes)
            if len(names_raw) != n_nodes:
                names = [str(i) for i in range(n_nodes)]
            else:
                names = [str(x) for x in names_raw]

            roles_raw = _metadata_list_from_graph_attr(g_data, "node_roles", n_nodes)
            if len(roles_raw) != n_nodes:
                roles = [_infer_staff_role_from_name(name) for name in names]
            else:
                roles = [_normalize_role_value(x, node_name=names[i]) for i, x in enumerate(roles_raw)]

            w_home = getattr(g_data, "node_ward_id", None)
            if torch.is_tensor(w_home):
                ward_id = [int(x) for x in w_home.detach().cpu().tolist()]
            elif w_home is None:
                ward_id = [0] * n_nodes
            else:
                try:
                    ward_id = [int(x) for x in list(w_home)]
                except Exception:
                    ward_id = [0] * n_nodes
            if len(ward_id) != n_nodes:
                ward_id = [0] * n_nodes

            w_cov = getattr(g_data, "node_ward_cover_count", None)
            if torch.is_tensor(w_cov):
                ward_cover = [max(1, int(x)) for x in w_cov.detach().cpu().tolist()]
            elif w_cov is None:
                ward_cover = [1] * n_nodes
            else:
                try:
                    ward_cover = [max(1, int(x)) for x in list(w_cov)]
                except Exception:
                    ward_cover = [1] * n_nodes
            if len(ward_cover) != n_nodes:
                ward_cover = [1] * n_nodes

            ward_ids_raw = _metadata_list_from_graph_attr(g_data, "node_ward_ids", n_nodes)
            if len(ward_ids_raw) != n_nodes:
                ward_ids = [str(ward_id[i]) for i in range(n_nodes)]
            else:
                ward_ids = [str(x) for x in ward_ids_raw]

            x_cpu = g_data.x.detach().cpu() if hasattr(g_data, "x") and torch.is_tensor(g_data.x) else None
            if x_cpu is not None and x_cpu.ndim == 2 and x_cpu.size(0) == n_nodes:
                state_signal, state_signal_name = _infer_state_signal_from_features(x_cpu)
            else:
                state_signal = [0.0] * n_nodes
                state_signal_name = "unknown"

            if len(state_signal) != n_nodes:
                state_signal = [0.0] * n_nodes
                state_signal_name = "unknown"

            names_per_graph.append(names)
            roles_per_graph.append(roles)
            ward_id_per_graph.append(ward_id)
            ward_ids_per_graph.append(ward_ids)
            ward_cover_per_graph.append(ward_cover)
            state_signal_per_graph.append(state_signal)
            state_signal_name_per_graph.append(state_signal_name)

        day_node_names.append(names_per_graph)
        day_node_roles.append(roles_per_graph)
        day_ward_id.append(ward_id_per_graph)
        day_ward_ids.append(ward_ids_per_graph)
        day_ward_cover_count.append(ward_cover_per_graph)
        day_node_state_signal.append(state_signal_per_graph)
        day_node_state_signal_name.append(state_signal_name_per_graph)

    H = torch.stack(day_embs, dim=1)
    action_features = _extract_action_features(batched_graphs, labels_dict, device=device)
    state_summary_features = _extract_state_summary_features(batched_graphs, labels_dict, device=device)
    y_hat = model.forward_from_day_embeddings(
        H,
        action_features=action_features,
        state_summary_features=state_summary_features,
    )
    return (
        y_hat,
        day_attn,
        day_batches,
        day_node_names,
        day_node_roles,
        day_ward_id,
        day_ward_ids,
        day_ward_cover_count,
        day_node_state_signal,
        day_node_state_signal_name,
    )



def _collect_fullgraph_attention_records(model, task, loader, device):
    records = []
    sample_counter = 0
    with torch.no_grad():
        for batched_graphs, labels_dict in loader:
            (
                y_hat,
                day_attn,
                day_batches,
                day_node_names,
                day_node_roles,
                day_ward_id,
                day_ward_ids,
                day_ward_cover_count,
                day_node_state_signal,
                day_node_state_signal_name,
            ) = _run_fullgraph_day_embeddings_and_attention(model, batched_graphs, labels_dict, device)
            B = y_hat.size(0)
            T = len(day_attn)

            label_targets = task.get_targets(batched_graphs, labels_dict).cpu()

            for b in range(B):
                sample_id = f"sample_{sample_counter:08d}"
                sample_counter += 1

                per_node = defaultdict(
                    lambda: {
                        "sum_attn": 0.0,
                        "count": 0,
                        "role": "unknown",
                        "ward_id": 0,
                        "ward_ids": "",
                        "ward_cover_count": 1,
                        "sum_state_signal": 0.0,
                        "state_signal_name": "unknown",
                    }
                )
                for t in range(T):
                    idx = (day_batches[t] == b).nonzero(as_tuple=False).view(-1).tolist()
                    names_b = day_node_names[t][b] if b < len(day_node_names[t]) else []
                    roles_b = day_node_roles[t][b] if b < len(day_node_roles[t]) else []
                    ward_id_b = day_ward_id[t][b] if b < len(day_ward_id[t]) else []
                    ward_ids_b = day_ward_ids[t][b] if b < len(day_ward_ids[t]) else []
                    ward_cover_b = day_ward_cover_count[t][b] if b < len(day_ward_cover_count[t]) else []
                    state_signal_b = day_node_state_signal[t][b] if b < len(day_node_state_signal[t]) else []
                    state_signal_name_b = day_node_state_signal_name[t][b] if b < len(day_node_state_signal_name[t]) else "unknown"

                    for local_graph_idx, local_idx in enumerate(idx):
                        node_name = str(names_b[local_graph_idx]) if local_graph_idx < len(names_b) else str(local_graph_idx)
                        rec = per_node[node_name]
                        rec["sum_attn"] += float(day_attn[t][local_idx].item())
                        rec["count"] += 1

                        role_val = roles_b[local_graph_idx] if local_graph_idx < len(roles_b) else _infer_staff_role_from_name(node_name)
                        rec["role"] = _normalize_role_value(role_val, node_name=node_name)

                        if local_graph_idx < len(ward_id_b):
                            rec["ward_id"] = int(ward_id_b[local_graph_idx])
                        if local_graph_idx < len(ward_ids_b):
                            rec["ward_ids"] = str(ward_ids_b[local_graph_idx])
                        if local_graph_idx < len(ward_cover_b):
                            rec["ward_cover_count"] = max(1, int(ward_cover_b[local_graph_idx]))
                        if local_graph_idx < len(state_signal_b):
                            rec["sum_state_signal"] += float(state_signal_b[local_graph_idx])
                        rec["state_signal_name"] = str(state_signal_name_b)

                label_item = label_targets[b]
                label_val = int(label_item.item()) if task.is_classification else float(label_item.item())
                pred_val = int(torch.argmax(y_hat[b]).item()) if task.is_classification else float(y_hat[b].cpu().item())

                for node_name, rec in per_node.items():
                    mean_attn = rec["sum_attn"] / max(1, rec["count"])
                    mean_state_signal = rec["sum_state_signal"] / max(1, rec["count"])
                    records.append(
                        {
                            "sample_id": sample_id,
                            "node_name": node_name,
                            "role": rec["role"],
                            "ward_id": rec["ward_id"],
                            "ward_ids": rec["ward_ids"],
                            "ward_cover_count": rec["ward_cover_count"],
                            "mean_attention": mean_attn,
                            "sample_count": rec["count"],
                            "label": label_val,
                            "pred": pred_val,
                            "mean_state_signal": mean_state_signal,
                            "state_signal_name": rec["state_signal_name"],
                        }
                    )
    return records


def _write_note(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_ward_ids_field(value: Any, fallback_home_ward: Optional[int] = None) -> List[int]:
    out: List[int] = []
    seen = set()
    if value is None:
        value = ""
    if isinstance(value, (list, tuple)):
        items = value
    else:
        s = str(value).strip()
        items = re.findall(r"-?\d+", s) if s else []
    for x in items:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    if fallback_home_ward is not None:
        try:
            home = int(fallback_home_ward)
            if home not in seen:
                out.insert(0, home)
        except Exception:
            pass
    return out


def _sanitize_record_ward_signature(ward_ids: List[int], home_ward: int, ward_cover_count: int) -> List[int]:
    home_ward = int(home_ward)
    cover = max(1, int(ward_cover_count))
    ordered = []
    seen = set()
    if home_ward not in seen:
        ordered.append(home_ward)
        seen.add(home_ward)
    for w in ward_ids:
        wi = int(w)
        if wi not in seen:
            ordered.append(wi)
            seen.add(wi)
    if len(ordered) > cover:
        ordered = ordered[:cover]
    return ordered


def _signature_tuple(wards: List[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(w) for w in wards))


def _ranked_labels(indices: List[int], scores: List[float], top_n: int = 8) -> List[int]:
    pairs = sorted(zip(indices, scores), key=lambda x: (-float(x[1]), int(x[0])))
    keep = {idx for idx, _ in pairs[: max(1, int(top_n))]}
    return [idx for idx in indices if idx in keep]


def _safe_corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if np.allclose(np.std(xa), 0.0) or np.allclose(np.std(ya), 0.0):
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _apply_publication_axes_style(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3)
    ax.grid(False)


def _canonical_staff_payload(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not observations:
        return {
            "home_ward": 0,
            "ward_cover_count": 1,
            "wards": [],
            "canonical_fraction": 1.0,
            "n_observations": 0,
            "n_unique_signatures": 0,
        }

    sig_counter: Counter = Counter()
    for obs in observations:
        sig_counter[obs["ward_signature"]] += 1

    canonical_signature, canonical_n = sorted(
        sig_counter.items(),
        key=lambda kv: (-int(kv[1]), len(kv[0]), kv[0]),
    )[0]

    canonical_obs = [obs for obs in observations if obs["ward_signature"] == canonical_signature]
    home_counter = Counter(int(obs["home_ward"]) for obs in canonical_obs)
    canonical_home = sorted(home_counter.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]

    cover_counter = Counter(int(obs["ward_cover_count"]) for obs in canonical_obs)
    canonical_cover = sorted(cover_counter.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]
    canonical_cover = max(1, min(int(canonical_cover), max(1, len(canonical_signature))))

    return {
        "home_ward": int(canonical_home),
        "ward_cover_count": int(canonical_cover),
        "wards": list(canonical_signature),
        "canonical_fraction": float(canonical_n) / float(max(1, len(observations))),
        "n_observations": int(len(observations)),
        "n_unique_signatures": int(len(sig_counter)),
    }


def _build_translational_publication_payload(records: List[Dict[str, Any]], task: BaseTask, top_k: int) -> Dict[str, Any]:
    del task
    by_ward = defaultdict(
        lambda: {
            "sum_attn": 0.0,
            "count": 0,
            "sample_labels": {},
            "sample_preds": {},
            "staff_records": 0,
            "patient_records": 0,
            "sum_state_signal": 0.0,
            "state_signal_records": 0,
        }
    )
    by_staff = defaultdict(
        lambda: {
            "sum_attn": 0.0,
            "count": 0,
            "labels": [],
            "observations": [],
        }
    )

    for r in records:
        sample_id = str(r.get("sample_id", ""))
        node_name = str(r.get("node_name", ""))
        role = str(r.get("role", "unknown"))
        ward_id = int(r.get("ward_id", 0))
        ward_cover_count = max(1, int(r.get("ward_cover_count", 1)))
        mean_attention = float(r.get("mean_attention", 0.0))
        label_val = float(r.get("label", 0.0))
        pred_val = float(r.get("pred", 0.0))
        mean_state_signal = float(r.get("mean_state_signal", 0.0))
        ward_ids_raw = _parse_ward_ids_field(r.get("ward_ids", ""), fallback_home_ward=ward_id)
        ward_ids = _sanitize_record_ward_signature(ward_ids_raw, home_ward=ward_id, ward_cover_count=ward_cover_count)
        ward_signature = _signature_tuple(ward_ids)

        bw = by_ward[ward_id]
        bw["sum_attn"] += mean_attention
        bw["count"] += 1
        bw["sample_labels"][sample_id] = label_val
        bw["sample_preds"][sample_id] = pred_val
        bw["sum_state_signal"] += mean_state_signal
        bw["state_signal_records"] += 1
        if _is_staff_role(role, node_name=node_name):
            bw["staff_records"] += 1
        else:
            bw["patient_records"] += 1

        if _is_staff_role(role, node_name=node_name):
            bs = by_staff[node_name]
            bs["sum_attn"] += mean_attention
            bs["count"] += 1
            bs["labels"].append(label_val)
            bs["observations"].append(
                {
                    "home_ward": int(ward_id),
                    "ward_cover_count": int(ward_cover_count),
                    "wards": list(ward_ids),
                    "ward_signature": ward_signature,
                }
            )

    total_state_signal = float(
        sum(float(rec["sum_state_signal"]) for rec in by_ward.values())
    )

    ward_rows = []
    for w in sorted(by_ward):
        rec = by_ward[w]
        mean_attn = rec["sum_attn"] / max(1, rec["count"])
        sample_labels = list(rec["sample_labels"].values())
        sample_preds = list(rec["sample_preds"].values())
        state_signal_total = float(rec["sum_state_signal"])
        state_signal_mean = state_signal_total / max(1, rec["state_signal_records"])
        state_signal_share = state_signal_total / max(total_state_signal, 1e-12)
        ward_rows.append(
            {
                "ward_id": int(w),
                "mean_attention": float(mean_attn),
                "target_association": float(np.mean(sample_labels)) if sample_labels else 0.0,
                "mean_pred": float(np.mean(sample_preds)) if sample_preds else 0.0,
                "node_records": int(rec["count"]),
                "n_samples_with_ward": int(len(sample_labels)),
                "staff_record_fraction": float(rec["staff_records"]) / max(1, rec["count"]),
                "state_signal_total": float(state_signal_total),
                "state_signal_mean": float(state_signal_mean),
                "state_signal_share": float(state_signal_share),
            }
        )

    staff_rows = []
    for node_name, rec in by_staff.items():
        mean_attn = rec["sum_attn"] / max(1, rec["count"])
        canonical = _canonical_staff_payload(rec["observations"])
        canonical_cover = max(1, int(canonical["ward_cover_count"]))
        bridge_score = float(mean_attn) * float(max(0, canonical_cover - 1))
        home_component = float(mean_attn) / float(canonical_cover)
        cross_component = float(mean_attn) * float(max(0, canonical_cover - 1)) / float(canonical_cover)
        staff_rows.append(
            {
                "node_name": str(node_name),
                "home_ward": int(canonical["home_ward"]),
                "ward_cover_count": int(canonical_cover),
                "wards": list(canonical["wards"]),
                "mean_attention": float(mean_attn),
                "bridge_score": float(bridge_score),
                "home_component": float(home_component),
                "cross_component": float(cross_component),
                "target_association": float(np.mean(rec["labels"])) if rec["labels"] else 0.0,
                "canonical_fraction": float(canonical["canonical_fraction"]),
                "n_observations": int(canonical["n_observations"]),
                "n_unique_signatures": int(canonical["n_unique_signatures"]),
            }
        )

    ward_rows = sorted(ward_rows, key=lambda d: (-d["mean_attention"], d["ward_id"]))
    staff_rows = sorted(
        staff_rows,
        key=lambda d: (-d["bridge_score"], -d["canonical_fraction"], d["node_name"]),
    )

    cross_edges = defaultdict(float)
    for row in staff_rows:
        wards = list(row.get("wards", []))
        if len(wards) < 2:
            continue
        comb_denom = max(1.0, len(wards) * (len(wards) - 1) / 2.0)
        edge_unit = float(row["cross_component"]) / float(comb_denom)
        for i in range(len(wards)):
            for j in range(i + 1, len(wards)):
                pair = tuple(sorted((int(wards[i]), int(wards[j]))))
                cross_edges[pair] += edge_unit

    edge_rows = sorted(
        [{"ward_a": int(a), "ward_b": int(b), "bridge_weight": float(w)} for (a, b), w in cross_edges.items()],
        key=lambda d: (-d["bridge_weight"], d["ward_a"], d["ward_b"]),
    )

    diagnostics_rows = []
    for row in staff_rows:
        diagnostics_rows.append(
            {
                "node_name": row["node_name"],
                "home_ward": row["home_ward"],
                "ward_cover_count": row["ward_cover_count"],
                "wards": row["wards"],
                "canonical_fraction": row["canonical_fraction"],
                "n_observations": row["n_observations"],
                "n_unique_signatures": row["n_unique_signatures"],
            }
        )

    state_signal_names = sorted(
        set(str(r.get("state_signal_name", "unknown")) for r in records if str(r.get("state_signal_name", "")).strip() != "")
    )
    if len(state_signal_names) == 1:
        state_signal_name = state_signal_names[0]
    elif len(state_signal_names) == 0:
        state_signal_name = "unknown"
    else:
        state_signal_name = "mixed_state_signal"

    return {
        "ward_rows": ward_rows,
        "staff_rows": staff_rows,
        "top_staff_rows": staff_rows[: max(1, int(top_k))],
        "edge_rows": edge_rows,
        "diagnostics_rows": diagnostics_rows,
        "state_signal_name": state_signal_name,
    }


def _save_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, set)):
                    out[key] = "|".join(str(x) for x in value)
                else:
                    out[key] = value
            writer.writerow(out)


def _draw_ward_influence_network(ax, ward_rows: List[Dict[str, Any]], edge_rows: List[Dict[str, Any]], state_signal_name: str):
    ax.set_title("A  Ward influence network", loc="left", fontsize=13, fontweight="bold")
    if not ward_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No ward records", ha="center", va="center", fontsize=11)
        return

    wards = [int(r["ward_id"]) for r in ward_rows]
    attn = np.asarray([float(r["mean_attention"]) for r in ward_rows], dtype=float)
    state_signal_share = np.asarray([float(r.get("state_signal_share", 0.0)) for r in ward_rows], dtype=float)

    n = len(wards)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {ward: np.array([np.cos(t), np.sin(t)]) for ward, t in zip(wards, theta)}

    edge_w = np.asarray([float(r["bridge_weight"]) for r in edge_rows], dtype=float) if edge_rows else np.asarray([], dtype=float)
    emax = float(edge_w.max()) if edge_w.size else 1.0
    for row in edge_rows:
        a = int(row["ward_a"])
        b = int(row["ward_b"])
        if a not in pos or b not in pos:
            continue
        p1 = pos[a]
        p2 = pos[b]
        w = float(row["bridge_weight"])
        frac = 0.15 + 0.85 * (w / max(emax, 1e-12))
        patch = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-",
            connectionstyle="arc3,rad=0.16",
            linewidth=0.8 + 4.0 * frac,
            color="#385170",
            alpha=0.12 + 0.48 * frac,
            zorder=1,
        )
        ax.add_patch(patch)

    sizes = 600 + 2600 * (attn / max(float(attn.max()), 1e-12))
    sc = ax.scatter(
        [pos[w][0] for w in wards],
        [pos[w][1] for w in wards],
        s=sizes,
        c=state_signal_share,
        cmap="YlGnBu",
        edgecolors="#18324a",
        linewidths=1.0,
        zorder=3,
    )
    for ward in _ranked_labels(wards, attn.tolist(), top_n=min(8, len(wards))):
        p = pos[ward]
        ax.text(p[0], p[1], f"W{ward}", ha="center", va="center", fontsize=10, fontweight="bold", color="#0b1f2a", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.40, 1.40)
    ax.axis("off")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"Ward share of total {state_signal_name.replace('_', ' ')}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.text(
        -1.48,
        -1.30,
        "Node size ∝ mean ward attention\nEdge width ∝ canonical cross-ward staff bridge weight",
        fontsize=8.5,
        ha="left",
        va="bottom",
        color="#334e68",
    )


def _draw_staff_ward_bipartite(ax, top_staff_rows: List[Dict[str, Any]], ward_rows: List[Dict[str, Any]]):
    ax.set_title("B  Staff–ward bridge map", loc="left", fontsize=13, fontweight="bold")
    if not top_staff_rows or not ward_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient staff/ward records", ha="center", va="center", fontsize=11)
        return

    ward_ids = sorted(int(r["ward_id"]) for r in ward_rows)
    staff_rows = top_staff_rows[: min(10, len(top_staff_rows))]

    y_staff = np.linspace(0.92, 0.08, len(staff_rows))
    y_ward = np.linspace(0.92, 0.08, len(ward_ids))
    x_staff = 0.18
    x_ward = 0.82

    attn_staff = np.asarray([float(r["mean_attention"]) for r in staff_rows], dtype=float)
    ward_attn_map = {int(r["ward_id"]): float(r["mean_attention"]) for r in ward_rows}
    ward_y_map = {ward: y for ward, y in zip(ward_ids, y_ward)}

    for srow, ys in zip(staff_rows, y_staff):
        wards = list(srow.get("wards", [])) or [int(srow.get("home_ward", 0))]
        for ward in wards:
            if ward not in ward_y_map:
                continue
            yw = ward_y_map[ward]
            width = 0.8 + 4.5 * float(srow["mean_attention"]) / max(float(attn_staff.max()), 1e-12)
            alpha = 0.10 + 0.18 * float(srow.get("canonical_fraction", 1.0))
            patch = FancyArrowPatch(
                (x_staff + 0.035, ys),
                (x_ward - 0.035, yw),
                arrowstyle="-",
                connectionstyle="arc3,rad=0.12",
                linewidth=width,
                color="#486581",
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(patch)

    for srow, ys in zip(staff_rows, y_staff):
        size = 180 + 900 * float(srow["mean_attention"]) / max(float(attn_staff.max()), 1e-12)
        edge_col = "#7b8794" if float(srow.get("canonical_fraction", 1.0)) < 0.75 else "white"
        ax.scatter([x_staff], [ys], s=size, color="#102a43", alpha=0.95, zorder=3, edgecolors=edge_col, linewidths=1.1)
        label = str(srow["node_name"])
        if float(srow.get("canonical_fraction", 1.0)) < 0.75:
            label += " *"
        ax.text(x_staff - 0.03, ys, label, ha="right", va="center", fontsize=9.3, color="#102a43")

    ward_attn_max = max(max(ward_attn_map.values()), 1e-12) if ward_attn_map else 1.0
    for ward in ward_ids:
        yw = ward_y_map[ward]
        size = 220 + 850 * ward_attn_map.get(ward, 0.0) / ward_attn_max
        ax.scatter([x_ward], [yw], s=size, color="#9fb3c8", alpha=1.0, zorder=3, edgecolors="#243b53", linewidths=0.9)
        ax.text(x_ward + 0.03, yw, f"W{ward}", ha="left", va="center", fontsize=9.3, color="#243b53")

    ax.text(x_staff, 0.99, "Top bridge staff", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#102a43")
    ax.text(x_ward, 0.99, "Wards", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#243b53")
    ax.text(0.02, 0.01, "* indicates unstable ward signature across the split", transform=ax.transAxes, fontsize=8.2, color="#52606d")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.axis("off")


def _draw_home_cross_diverging(ax, top_staff_rows: List[Dict[str, Any]]):
    ax.set_title("C  Coverage-based home vs cross-ward decomposition", loc="left", fontsize=13, fontweight="bold")
    if not top_staff_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No staff rows", ha="center", va="center", fontsize=11)
        return

    rows = top_staff_rows[: min(10, len(top_staff_rows))]
    labels = [str(r["node_name"]) for r in rows][::-1]
    home = np.asarray([float(r["home_component"]) for r in rows][::-1], dtype=float)
    cross = np.asarray([float(r["cross_component"]) for r in rows][::-1], dtype=float)
    stability = np.asarray([float(r.get("canonical_fraction", 1.0)) for r in rows][::-1], dtype=float)
    y = np.arange(len(labels))

    ax.barh(y, -home, color="#98c1d9", edgecolor="white", linewidth=0.6, label="Home share")
    ax.barh(y, cross, color="#3d5a80", edgecolor="white", linewidth=0.6, label="Cross-ward share")
    ax.axvline(0.0, color="#243b53", linewidth=0.8)
    ax.scatter(np.zeros_like(y), y, s=16 + 50 * stability, color="#102a43", alpha=0.65, zorder=4, label="Signature stability")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Attention contribution", fontsize=10)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    _apply_publication_axes_style(ax)
    ax.text(0.02, 0.02, "Coverage-count decomposition; descriptive, not causal", transform=ax.transAxes, fontsize=8.0, color="#52606d")


def _draw_ward_target_scatter(ax, ward_rows: List[Dict[str, Any]], state_signal_name: str):
    ax.set_title("D  Attention–burden concordance", loc="left", fontsize=13, fontweight="bold")
    if not ward_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No ward rows", ha="center", va="center", fontsize=11)
        return

    x = np.asarray([float(r["mean_attention"]) for r in ward_rows], dtype=float)
    y = np.asarray([float(r.get("state_signal_share", 0.0)) for r in ward_rows], dtype=float)
    s = np.asarray([float(r["n_samples_with_ward"]) for r in ward_rows], dtype=float)
    ward_ids = [int(r["ward_id"]) for r in ward_rows]

    sizes = 90 + 420 * (s / max(float(s.max()), 1e-12))
    sc = ax.scatter(x, y, s=sizes, c=y, cmap="YlGnBu", edgecolors="#274c77", linewidths=0.8, alpha=0.92)

    if len(x) >= 2 and not np.allclose(np.std(x), 0.0):
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, color="#102a43", linewidth=1.5, linestyle="-")

    for ward in _ranked_labels(ward_ids, x.tolist(), top_n=min(6, len(ward_ids))):
        idx = ward_ids.index(ward)
        ax.annotate(f"W{ward}", (x[idx], y[idx]), fontsize=8.5, xytext=(4, 4), textcoords="offset points", color="#102a43")

    corr = _safe_corr(x.tolist(), y.tolist())
    corr_txt = "NA" if np.isnan(corr) else f"{corr:.2f}"
    ax.text(0.03, 0.97, f"Pearson r = {corr_txt}", transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#102a43")
    ax.set_xlabel("Mean ward attention", fontsize=10)
    ax.set_ylabel(f"Ward share of total {state_signal_name.replace('_', ' ')}", fontsize=10)
    _apply_publication_axes_style(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(f"Ward share of total {state_signal_name.replace('_', ' ')}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _draw_ward_lollipop(ax, ward_rows: List[Dict[str, Any]], state_signal_name: str):
    ax.set_title("E  Ward attribution ranking", loc="left", fontsize=13, fontweight="bold")
    if not ward_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No ward rows", ha="center", va="center", fontsize=11)
        return

    rows = sorted(ward_rows, key=lambda d: (-float(d["mean_attention"]), int(d["ward_id"])))[: min(10, len(ward_rows))]
    rows = rows[::-1]
    y = np.arange(len(rows))
    x = np.asarray([float(r["mean_attention"]) for r in rows], dtype=float)
    c = np.asarray([float(r.get("state_signal_share", 0.0)) for r in rows], dtype=float)
    labels = [f"W{int(r['ward_id'])}" for r in rows]

    ax.hlines(y, xmin=0.0, xmax=x, color="#9fb3c8", linewidth=2.0)
    sc = ax.scatter(x, y, s=110, c=c, cmap="YlGnBu", edgecolors="#243b53", linewidths=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean attention", fontsize=10)
    _apply_publication_axes_style(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(f"Ward share of total {state_signal_name.replace('_', ' ')}", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _draw_staff_bridge_dotplot(ax, top_staff_rows: List[Dict[str, Any]]):
    ax.set_title("F  Staff bridge ranking", loc="left", fontsize=13, fontweight="bold")
    if not top_staff_rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "No staff rows", ha="center", va="center", fontsize=11)
        return

    rows = top_staff_rows[: min(10, len(top_staff_rows))][::-1]
    y = np.arange(len(rows))
    x = np.asarray([float(r["bridge_score"]) for r in rows], dtype=float)
    size = np.asarray([float(r["ward_cover_count"]) for r in rows], dtype=float)
    alpha = np.asarray([float(r.get("canonical_fraction", 1.0)) for r in rows], dtype=float)
    labels = [str(r["node_name"]) for r in rows]

    ax.hlines(y, xmin=0.0, xmax=x, color="#d9e2ec", linewidth=2.0)
    ax.scatter(
        x,
        y,
        s=80 + 80 * size,
        color="#102a43",
        alpha=np.clip(alpha, 0.35, 0.95),
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Bridge score", fontsize=10)
    _apply_publication_axes_style(ax)
    ax.text(0.02, 0.02, "Dot opacity ∝ ward-signature stability", transform=ax.transAxes, fontsize=8.0, color="#52606d")


def _save_publication_translational_microgrid(
    out_root: Path,
    split_name: str,
    ward_rows: List[Dict[str, Any]],
    top_staff_rows: List[Dict[str, Any]],
    edge_rows: List[Dict[str, Any]],
    state_signal_name: str,
) -> None:
    fig = plt.figure(figsize=(16.5, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    _draw_ward_influence_network(ax_a, ward_rows, edge_rows, state_signal_name)
    _draw_staff_ward_bipartite(ax_b, top_staff_rows, ward_rows)
    _draw_home_cross_diverging(ax_c, top_staff_rows)
    _draw_ward_target_scatter(ax_d, ward_rows, state_signal_name)
    _draw_ward_lollipop(ax_e, ward_rows, state_signal_name)
    _draw_staff_bridge_dotplot(ax_f, top_staff_rows)

    fig.suptitle(
        f"Translational attribution microgrid — {split_name}",
        fontsize=17,
        fontweight="bold",
        x=0.02,
        y=1.01,
        ha="left",
    )

    png_path = out_root / f"translational_microgrid_{split_name.lower()}.png"
    pdf_path = out_root / f"translational_microgrid_{split_name.lower()}.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_translational_figures_fullgraph(
    model,
    task,
    loader,
    device,
    out_dir: str,
    split_name: str,
    translational_top_k: int,
    trained_with_neighbor_sampling: bool,
    fullgraph_attribution_pass: bool,
):
    out_root = Path(out_dir) / "translational_figures"
    out_root.mkdir(parents=True, exist_ok=True)
    records = _collect_fullgraph_attention_records(model, task, loader, device)
    if len(records) == 0:
        _write_note(out_root / f"translational_note_{split_name.lower()}.txt", f"No translational records available for {split_name}.")
        return

    payload = _build_translational_publication_payload(
        records=records,
        task=task,
        top_k=max(1, int(translational_top_k)),
    )

    ward_rows = payload["ward_rows"]
    staff_rows = payload["staff_rows"]
    top_staff_rows = payload["top_staff_rows"]
    edge_rows = payload["edge_rows"]
    diagnostics_rows = payload["diagnostics_rows"]
    state_signal_name = str(payload.get("state_signal_name", "unknown"))

    n_unstable = sum(1 for r in diagnostics_rows if float(r.get("canonical_fraction", 1.0)) < 0.75)
    note = [
        f"split={split_name}",
        f"trained_with_neighbor_sampling={bool(trained_with_neighbor_sampling)}",
        f"fullgraph_attribution_pass={bool(fullgraph_attribution_pass)}",
        "Interpretation: post hoc model-emphasis summary under full-graph inference; not a causal effect estimate.",
        f"Ward color in panel A encodes ward share of total {state_signal_name.replace('_', ' ')} across the split.",
        f"Panel D relates mean ward attention to ward share of total {state_signal_name.replace('_', ' ')} across the split.",
        f"Panel E colors wards by the same split-level burden-share quantity used in panels A and D.",
        "Staff bridge panels use a canonical per-staff ward signature (most frequent observed ward-set across the split) to avoid union inflation.",
        "Home vs cross-ward decomposition is coverage-count based and descriptive only.",
        f"unstable_staff_signatures={n_unstable}",
    ]
    _write_note(out_root / f"translational_note_{split_name.lower()}.txt", "\n".join(note) + "\n")

    _save_csv_rows(
        out_root / f"ward_attribution_{split_name.lower()}.csv",
        ["ward_id", "mean_attention", "target_association", "mean_pred", "node_records", "n_samples_with_ward", "staff_record_fraction", "state_signal_total", "state_signal_mean", "state_signal_share"],
        ward_rows,
    )
    _save_csv_rows(
        out_root / f"staff_bridge_{split_name.lower()}.csv",
        ["node_name", "home_ward", "ward_cover_count", "wards", "mean_attention", "bridge_score", "target_association", "canonical_fraction", "n_observations", "n_unique_signatures"],
        top_staff_rows,
    )
    _save_csv_rows(
        out_root / f"ward_target_proxy_{split_name.lower()}.csv",
        ["ward_id", "mean_attention", "target_association", "mean_pred", "node_records", "n_samples_with_ward", "staff_record_fraction", "state_signal_total", "state_signal_mean", "state_signal_share"],
        ward_rows,
    )
    _save_csv_rows(
        out_root / f"home_cross_{split_name.lower()}.csv",
        ["node_name", "home_ward", "ward_cover_count", "mean_attention", "home_component", "cross_component", "bridge_score", "wards", "canonical_fraction", "n_observations", "n_unique_signatures"],
        top_staff_rows,
    )
    _save_csv_rows(
        out_root / f"ward_network_edges_{split_name.lower()}.csv",
        ["ward_a", "ward_b", "bridge_weight"],
        edge_rows,
    )
    _save_csv_rows(
        out_root / f"staff_signature_diagnostics_{split_name.lower()}.csv",
        ["node_name", "home_ward", "ward_cover_count", "wards", "canonical_fraction", "n_observations", "n_unique_signatures"],
        diagnostics_rows,
    )
    _save_csv_rows(
        out_root / f"fullgraph_attention_records_{split_name.lower()}.csv",
        ["sample_id", "node_name", "role", "ward_id", "ward_ids", "ward_cover_count", "mean_attention", "sample_count", "label", "pred", "mean_state_signal", "state_signal_name"],
        records,
    )

    _save_publication_translational_microgrid(
        out_root=out_root,
        split_name=split_name,
        ward_rows=ward_rows,
        top_staff_rows=top_staff_rows,
        edge_rows=edge_rows,
        state_signal_name=state_signal_name,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument(
        "--test_folder",
        type=str,
        default=None,
        help=(
            "Optional explicit path to the external test dataset folder. "
            "If omitted, the script will look for 'synthetic_amr_graphs_test' "
            "or 'synthetic_amr_graphs_test_pt' beside the training data folder."
        ),
    )
    parser.add_argument("--task", type=str, required=True)

    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--sliding_step", type=int, default=1)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--sage_layers", type=int, default=3)
    parser.add_argument("--use_cls", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cpu_threads", type=int, default=max(1, (os.cpu_count() or 1)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--max_neighbors", type=int, default=20)

    parser.add_argument("--neighbor_sampling", type=str2bool, default=None)
    parser.add_argument("--num_neighbors", type=str, default=None)
    parser.add_argument("--seed_count", type=int, default=None)
    parser.add_argument("--seed_strategy", type=str, default=None)
    parser.add_argument("--seed_batch_size", type=int, default=None)
    parser.add_argument("--max_sub_batches", type=int, default=None)

    # RESTORED: Attention heatmap controls
    parser.add_argument("--attn_top_k", type=int, default=50, help="Top-K nodes to show in attention heatmaps.")
    parser.add_argument(
        "--attn_rank_by",
        type=str,
        default="abs_diff",
        choices=["abs_diff", "mean"],
        help="Rank nodes for heatmap by absolute difference between groups or by mean attention.",
    )

    parser.add_argument("--emit_translational_figures", type=str2bool, default=True, help="Export translational attribution figures.")
    parser.add_argument("--fullgraph_attribution_pass", type=str2bool, default=True, help="When training uses neighbor sampling, export translational figures and attention heatmaps from a separate full-graph inference pass.")
    parser.add_argument("--translational_top_k", type=int, default=20, help="Top-K staff/nodes to include in translational attribution rankings.")
    parser.add_argument("--use_action_conditioning", type=str2bool, default=False, help="If true, condition the temporal predictor on per-sample action features from the causal-policy dataset.")
    parser.add_argument("--action_hidden_dim", type=int, default=32, help="Hidden dimension for projected action features when action conditioning is enabled.")
    parser.add_argument("--action_interaction_hidden_dim", type=int, default=128, help="Hidden dimension for the nonlinear state-action interaction head when action conditioning is enabled.")
    parser.add_argument("--action_interaction_dropout", type=float, default=0.1, help="Dropout used inside the nonlinear state-action interaction head.")
    parser.add_argument("--aux_policy_loss", type=str2bool, default=True, help="If true, add an auxiliary continuous-oracle loss for action-conditioned policy-manifest classification tasks when the matching y_h* target is available.")
    parser.add_argument("--aux_policy_loss_weight", type=float, default=0.25, help="Weight for the auxiliary continuous-oracle policy loss.")
    parser.add_argument("--aux_policy_target_name", type=str, default="", help="Optional explicit continuous target name for the auxiliary policy loss, e.g. y_h7_endog_share.")
    parser.add_argument("--pairwise_policy_ranking_loss", type=str2bool, default=True, help="If true, add a within-state pairwise and listwise ranking loss over action rows using the continuous oracle target.")
    parser.add_argument("--pairwise_policy_ranking_weight", type=float, default=2.0, help="Weight for the within-state pairwise and listwise policy ranking loss.")
    parser.add_argument("--pairwise_policy_margin", type=float, default=0.05, help="Margin for the within-state pairwise ranking loss.")
    parser.add_argument("--pairwise_policy_min_target_gap", type=float, default=0.01, help="Minimum oracle-target gap required to include an action pair in the ranking loss.")
    parser.add_argument("--oracle_best_action_loss", type=str2bool, default=True, help="If true, add a direct within-state softmax cross-entropy loss over candidate actions using the same utility score used for policy selection.")
    parser.add_argument("--oracle_best_action_loss_weight", type=float, default=1.0, help="Weight for the within-state oracle-best-action softmax cross-entropy loss.")
    parser.add_argument("--require_complete_action_set", type=str2bool, default=True, help="If true in policy-manifest mode, drop any state whose observed action rows do not exactly match the global expected action menu.")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--use_task_hparams", action="store_true")

    parser.add_argument(
        "--train_model",
        type=str2bool,
        default=True,
        help="true/false: if false, skip training and only run evaluation using a saved model.",
    )

    parser.add_argument(
        "--require_pt_metadata",
        type=str2bool,
        default=True,
        help="If true, require Data.sim_id and Data.day in all .pt files (recommended).",
    )
    parser.add_argument(
        "--fail_on_noncontiguous",
        type=str2bool,
        default=True,
        help="If true, raise if any non-contiguous temporal windows are detected.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="training_outputs",
        help="Directory for all outputs (plots + trained_model.pt). Default: training_outputs",
    )

    parser.add_argument("--early_stopping", type=str2bool, default=True)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--save_best_only", type=str2bool, default=True)
    parser.add_argument("--lr_scheduler_on_plateau", type=str2bool, default=False)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)

    args = parser.parse_args()
    args.cpu_threads = configure_cpu_threads(args.cpu_threads)
    args.data_folder = os.path.abspath(args.data_folder)
    if args.test_folder is not None:
        args.test_folder = os.path.abspath(args.test_folder)

    out_dir = os.path.abspath(str(args.out_dir))

    # Script-level defaults (applied if still None after parsing/task override)
    if args.neighbor_sampling is None:
        args.neighbor_sampling = False
    if args.num_neighbors is None:
        args.num_neighbors = "15,10"
    if args.seed_count is None:
        args.seed_count = 256
    if args.seed_strategy is None:
        args.seed_strategy = "random"
    if args.seed_batch_size is None:
        args.seed_batch_size = 64
    if args.max_sub_batches is None:
        args.max_sub_batches = 4

    args.num_neighbors = parse_num_neighbors(args.num_neighbors)
    if len(args.num_neighbors) == 0:
        args.num_neighbors = [15, 10]

    dataset = TemporalGraphDataset(
        folder=args.data_folder,
        T=args.T,
        sliding_step=args.sliding_step,
        prefer_pt_metadata=True,
        require_pt_metadata=bool(args.require_pt_metadata),
        fail_on_noncontiguous=bool(args.fail_on_noncontiguous),
    )

    # -------------------------------------------------------------------------
    # Train/Val split
    # -------------------------------------------------------------------------
    manifest_test_indices: List[int] = []
    policy_expected_action_ids: List[str] = []
    policy_split_filter_summary: Dict[str, Any] = {}
    if getattr(dataset, "use_policy_manifest", False):
        train_indices = []
        val_indices = []
        split_counts: Dict[str, int] = defaultdict(int)

        for i, sample in enumerate(getattr(dataset, "samples", [])):
            row_meta = sample.get("row_meta", {})
            split_raw = str(row_meta.get("split", "")).strip().lower()
            split_norm = "validation" if split_raw == "val" else split_raw
            split_counts[split_norm] += 1
            if split_norm == "train":
                train_indices.append(i)
            elif split_norm == "validation":
                val_indices.append(i)
            elif split_norm == "test":
                manifest_test_indices.append(i)

        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)
        manifest_test_indices = sorted(manifest_test_indices)

        policy_expected_action_ids = _expected_policy_action_ids(dataset)

        train_indices, dropped_train_states, policy_expected_action_ids = _filter_policy_manifest_indices_by_complete_action_set(
            dataset=dataset,
            indices=train_indices,
            split_name="train",
            require_complete_action_set=bool(args.require_complete_action_set),
        )
        val_indices, dropped_val_states, _ = _filter_policy_manifest_indices_by_complete_action_set(
            dataset=dataset,
            indices=val_indices,
            split_name="validation",
            require_complete_action_set=bool(args.require_complete_action_set),
        )
        manifest_test_indices, dropped_test_states, _ = _filter_policy_manifest_indices_by_complete_action_set(
            dataset=dataset,
            indices=manifest_test_indices,
            split_name="test",
            require_complete_action_set=bool(args.require_complete_action_set),
        )

        policy_split_filter_summary = {
            "expected_action_ids": list(policy_expected_action_ids),
            "expected_n_actions": int(len(policy_expected_action_ids)),
            "dropped_train_states": dropped_train_states,
            "dropped_validation_states": dropped_val_states,
            "dropped_test_states": dropped_test_states,
        }

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(
                "Policy-manifest mode requires non-empty train and val/validation splits in policy_manifest.csv "
                "after complete-action-set filtering. "
                f"Found train={len(train_indices)} validation={len(val_indices)} test={len(manifest_test_indices)} "
                f"for data_folder={args.data_folder}. "
                f"Expected action_ids={policy_expected_action_ids}."
            )

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        train_loader = _make_train_loader(
            dataset,
            train_set,
            args.batch_size,
            expected_action_ids=policy_expected_action_ids,
        )
        val_loader = _make_eval_loader(
            val_set,
            args.batch_size,
            expected_action_ids=policy_expected_action_ids,
        )
        print(
            f"✅ Policy-manifest split: train={len(train_indices)} | "
            f"validation={len(val_indices)} | test={len(manifest_test_indices)} | "
            f"expected_actions={len(policy_expected_action_ids)}"
        )
        print(
            f"✅ Complete-action-set filtering: dropped_train_states={len(dropped_train_states)} | "
            f"dropped_validation_states={len(dropped_val_states)} | "
            f"dropped_test_states={len(dropped_test_states)}"
        )
    else:
        val_ratio = 0.2
        sim_cache: Dict[str, str] = {}
        sim_to_indices: Dict[str, List[int]] = {}

        for i, fnames in enumerate(dataset.groups):
            sim_id = _get_window_sim_id(dataset, fnames, sim_cache)
            sim_to_indices.setdefault(sim_id, []).append(i)

        sim_ids = sorted(sim_to_indices.keys())
        if len(sim_ids) < 2:
            raise ValueError(
                "Leakage-safe train/validation split requires at least two distinct trajectories "
                f"(distinct sim_id values). Found {len(sim_ids)} trajectory in {args.data_folder}. "
                "Refusing to fall back to window-level random splitting because overlapping temporal windows "
                "from the same trajectory would leak information between train and validation."
            )

        rng = np.random.RandomState(args.split_seed)
        rng.shuffle(sim_ids)

        n_val_sims = int(round(len(sim_ids) * val_ratio))
        n_val_sims = max(1, min(len(sim_ids) - 1, n_val_sims))

        val_sims = set(sim_ids[:n_val_sims])
        train_indices = []
        val_indices = []

        for sid, idxs in sim_to_indices.items():
            if sid in val_sims:
                val_indices.extend(idxs)
            else:
                train_indices.extend(idxs)

        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(
                "Trajectory-level split produced an empty train or validation partition. "
                f"train_windows={len(train_indices)} val_windows={len(val_indices)} seed={args.split_seed}. "
                "Refusing to fall back to window-level random splitting because that would leak overlapping "
                "windows from the same trajectory across the split."
            )

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        train_loader = _make_train_loader(dataset, train_set, args.batch_size)
        val_loader = _make_eval_loader(val_set, args.batch_size)

    # -------------------------------------------------------------------------
    # Test set resolution
    # -------------------------------------------------------------------------
    test_loader = None
    test_dataset = None
    chosen_test_folder = None

    if getattr(dataset, "use_policy_manifest", False):
        if len(manifest_test_indices) > 0:
            test_set = Subset(dataset, manifest_test_indices)
            test_loader = _make_eval_loader(
                test_set,
                args.batch_size,
                expected_action_ids=policy_expected_action_ids,
            )
            chosen_test_folder = args.data_folder
            print(
                f"✅ Loaded test split from policy manifest in '{args.data_folder}' "
                f"with {len(manifest_test_indices)} windows."
            )
        else:
            print("⚠️ Policy-manifest dataset has no test split rows; test evaluation will be skipped.")
    else:
        run_base_dir = os.path.dirname(os.path.abspath(args.data_folder))
        auto_test_folder = os.path.abspath(os.path.join(run_base_dir, "synthetic_amr_graphs_test"))
        auto_alt_test_folder = os.path.abspath(os.path.join(run_base_dir, "synthetic_amr_graphs_test_pt"))

        candidate_test_folders: List[str] = []
        if args.test_folder is not None:
            candidate_test_folders.append(args.test_folder)
        candidate_test_folders.extend([auto_test_folder, auto_alt_test_folder])

        seen_test_folders = set()
        deduped_candidate_test_folders: List[str] = []
        for candidate in candidate_test_folders:
            if candidate not in seen_test_folders:
                deduped_candidate_test_folders.append(candidate)
                seen_test_folders.add(candidate)

        for candidate in deduped_candidate_test_folders:
            if os.path.isdir(candidate):
                chosen_test_folder = candidate
                break

        if chosen_test_folder is not None:
            test_dataset = TemporalGraphDataset(
                folder=chosen_test_folder,
                T=args.T,
                sliding_step=args.sliding_step,
                prefer_pt_metadata=True,
                require_pt_metadata=bool(args.require_pt_metadata),
                fail_on_noncontiguous=bool(args.fail_on_noncontiguous),
            )
            test_loader = _make_eval_loader(test_dataset, args.batch_size)
            if args.test_folder is not None and os.path.abspath(chosen_test_folder) == os.path.abspath(args.test_folder):
                print(
                    f"✅ Loaded test dataset from explicit --test_folder '{chosen_test_folder}' "
                    f"with {len(test_dataset)} windows."
                )
            else:
                print(
                    f"✅ Loaded test dataset from auto-detected folder '{chosen_test_folder}' "
                    f"with {len(test_dataset)} windows."
                )
        else:
            if args.test_folder is not None:
                print(
                    f"⚠️ Explicit test dataset folder '{args.test_folder}' was not found. "
                    f"Also checked auto-detected locations '{auto_test_folder}' and "
                    f"'{auto_alt_test_folder}'. Test evaluation will be skipped."
                )
            else:
                print(
                    f"⚠️ Test dataset folder not found at '{auto_test_folder}' "
                    f"(or '{auto_alt_test_folder}'). Test evaluation will be skipped."
                )

    in_channels, edge_dim = infer_feature_dims(dataset)
    action_feature_dim = infer_action_feature_dim(dataset) if bool(args.use_action_conditioning) else 0
    state_summary_feature_dim = infer_state_summary_feature_dim(dataset) if bool(args.use_action_conditioning) else 0
    if bool(args.use_action_conditioning) and action_feature_dim <= 0:
        raise ValueError(
            "--use_action_conditioning=true but no action_features attribute was found in the dataset. "
            "Convert the causal-policy GraphML files with the patched convert_to_pt.py first."
        )

    # -------------------------------------------------------------------------
    # Task resolution
    # -------------------------------------------------------------------------
    if args.task in TASK_REGISTRY:
        task_obj = TASK_REGISTRY[args.task]
        if isinstance(task_obj, BaseTask):
            task = task_obj
        elif isinstance(task_obj, type):
            task = task_obj()
        elif callable(task_obj):
            task = task_obj()
        else:
            raise TypeError(
                f"TASK_REGISTRY['{args.task}'] must be a BaseTask instance or a callable/class returning one; "
                f"got type={type(task_obj)}"
            )
    else:
        try:
            task = get_task(args.task)
        except Exception as e:
            raise ValueError(
                f"Unknown task '{args.task}'. Available registry keys: {list(TASK_REGISTRY.keys())}. "
                f"Also supports horizonised names like '<base>_h<H>' for supported bases. Error: {e}"
            ) from e

    _validate_task_labels(task, args.data_folder, dataset=dataset)

    # -------------------------------------------------------------------------
    # Task overrides (but do NOT stomp user-provided sampling flags)
    # -------------------------------------------------------------------------
    if args.use_task_hparams:
        model_cfg = getattr(task, "model_config", {}) or {}
        train_cfg = getattr(task, "train_config", {}) or {}

        args.hidden = model_cfg.get("hidden_channels", args.hidden)
        args.heads = model_cfg.get("heads", args.heads)
        args.dropout = model_cfg.get("dropout", args.dropout)
        args.transformer_layers = model_cfg.get("transformer_layers", args.transformer_layers)
        args.sage_layers = model_cfg.get("sage_layers", args.sage_layers)
        args.use_cls = model_cfg.get("use_cls_token", args.use_cls)

        args.batch_size = train_cfg.get("batch_size", args.batch_size)
        args.epochs = train_cfg.get("epochs", args.epochs)
        args.lr = train_cfg.get("lr", args.lr)
        args.max_neighbors = train_cfg.get("max_neighbors", args.max_neighbors)

        cfg_nn = train_cfg.get("num_neighbors", None)
        if cfg_nn is not None and args.num_neighbors == [15, 10]:
            args.num_neighbors = parse_num_neighbors(cfg_nn) if isinstance(cfg_nn, str) else list(cfg_nn)
        
        if train_cfg.get("seed_count", None) is not None and args.seed_count == 256:
            args.seed_count = int(train_cfg["seed_count"])
        
        if train_cfg.get("seed_strategy", None) is not None and args.seed_strategy == "random":
            args.seed_strategy = str(train_cfg["seed_strategy"])
        
        if train_cfg.get("seed_batch_size", None) is not None and args.seed_batch_size == 64:
            args.seed_batch_size = int(train_cfg["seed_batch_size"])
        
        if train_cfg.get("max_sub_batches", None) is not None and args.max_sub_batches == 4:
            args.max_sub_batches = int(train_cfg["max_sub_batches"])

        train_loader = _make_train_loader(
            dataset,
            train_set,
            args.batch_size,
            expected_action_ids=policy_expected_action_ids if getattr(dataset, "use_policy_manifest", False) else None,
        )
        val_loader = _make_eval_loader(
            val_set,
            args.batch_size,
            expected_action_ids=policy_expected_action_ids if getattr(dataset, "use_policy_manifest", False) else None,
        )

        if getattr(dataset, "use_policy_manifest", False):
            if len(manifest_test_indices) > 0:
                test_set = Subset(dataset, manifest_test_indices)
                test_loader = _make_eval_loader(
                    test_set,
                    args.batch_size,
                    expected_action_ids=policy_expected_action_ids,
                )
        elif test_dataset is not None:
            test_loader = _make_eval_loader(test_dataset, args.batch_size)

    output_activation = str(
        getattr(task, "output_activation", "identity" if task.is_classification else "softplus")
    ).lower().strip()
    use_softplus = (output_activation == "softplus")

    model = AMRDyGFormer(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        edge_dim=edge_dim,
        heads=args.heads,
        T=args.T,
        dropout=args.dropout,
        use_cls_token=args.use_cls,
        n_outputs=task.out_dim,
        n_layers=args.transformer_layers,
        sage_layers=args.sage_layers,
        use_softplus=use_softplus,
        output_activation=output_activation,
        action_feature_dim=action_feature_dim,
        action_hidden_dim=int(args.action_hidden_dim),
        use_action_conditioning=bool(args.use_action_conditioning),
        action_interaction_hidden_dim=int(args.action_interaction_hidden_dim),
        action_interaction_dropout=float(args.action_interaction_dropout),
        state_summary_feature_dim=int(state_summary_feature_dim),
        state_summary_hidden_dim=int(args.action_hidden_dim),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if bool(args.lr_scheduler_on_plateau):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.lr_scheduler_factor),
            patience=int(args.lr_scheduler_patience),
            min_lr=float(args.lr_scheduler_min_lr),
        )

    import shutil

    # Only wipe outputs when we are actually training; otherwise preserve the saved model.
    if args.train_model:
        shutil.rmtree(out_dir, ignore_errors=True)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "trained_model.pt")

    train_losses: List[float] = []
    val_losses: List[float] = []

    print(f"DT_PROGRESS_META epochs={args.epochs}", flush=True)
    print(f"DT_OUT_DIR {out_dir}", flush=True)
    print(f"DT_CPU_THREADS {int(args.cpu_threads)}", flush=True)

    if args.neighbor_sampling:
        print(
            f"✅ Neighbor sampling ENABLED: num_neighbors={args.num_neighbors} | "
            f"seed_count={args.seed_count} | seed_strategy={args.seed_strategy} | "
            f"seed_batch_size={args.seed_batch_size} | max_sub_batches={args.max_sub_batches}"
        )
    else:
        print(f"✅ Neighbor sampling DISABLED: using legacy max_neighbors={args.max_neighbors} edge-thinning.")

    best_val_loss = None
    best_epoch = None
    epochs_completed = 0
    stopped_early = False
    epochs_without_improvement = 0

    if args.train_model:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model=model,
                task=task,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                max_neighbors=args.max_neighbors,
                neighbor_sampling=args.neighbor_sampling,
                num_neighbors=args.num_neighbors,
                seed_count=args.seed_count,
                seed_strategy=args.seed_strategy,
                max_sub_batches=args.max_sub_batches,
                seed_batch_size=args.seed_batch_size,
                task_name=task.name,
                aux_policy_loss=bool(args.aux_policy_loss),
                aux_policy_loss_weight=float(args.aux_policy_loss_weight),
                aux_policy_target_name=str(args.aux_policy_target_name),
                pairwise_policy_ranking_loss=bool(args.pairwise_policy_ranking_loss),
                pairwise_policy_ranking_weight=float(args.pairwise_policy_ranking_weight),
                pairwise_policy_margin=float(args.pairwise_policy_margin),
                pairwise_policy_min_target_gap=float(args.pairwise_policy_min_target_gap),
                oracle_best_action_loss=bool(args.oracle_best_action_loss),
                oracle_best_action_loss_weight=float(args.oracle_best_action_loss_weight),
            )
            val_loss, val_metrics, _, _, _ = eval_epoch(
                model=model,
                task=task,
                loader=val_loader,
                device=device,
                max_neighbors=args.max_neighbors,
                neighbor_sampling=args.neighbor_sampling,
                num_neighbors=args.num_neighbors,
                seed_count=args.seed_count,
                seed_strategy=args.seed_strategy,
                max_sub_batches=args.max_sub_batches,
                seed_batch_size=args.seed_batch_size,
                task_name=task.name,
                aux_policy_loss=bool(args.aux_policy_loss),
                aux_policy_loss_weight=float(args.aux_policy_loss_weight),
                aux_policy_target_name=str(args.aux_policy_target_name),
                pairwise_policy_ranking_loss=bool(args.pairwise_policy_ranking_loss),
                pairwise_policy_ranking_weight=float(args.pairwise_policy_ranking_weight),
                pairwise_policy_margin=float(args.pairwise_policy_margin),
                pairwise_policy_min_target_gap=float(args.pairwise_policy_min_target_gap),
                oracle_best_action_loss=bool(args.oracle_best_action_loss),
                oracle_best_action_loss_weight=float(args.oracle_best_action_loss_weight),
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epochs_completed = epoch

            if scheduler is not None:
                scheduler.step(float(val_loss))

            improved = False
            if best_val_loss is None or float(val_loss) < float(best_val_loss) - float(args.min_delta):
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                epochs_without_improvement = 0
                improved = True
                if bool(args.save_best_only):
                    torch.save(model.state_dict(), model_path)
            else:
                epochs_without_improvement += 1

            metrics_str = " | ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
            if metrics_str:
                metrics_str = " | " + metrics_str

            lr_now = optimizer.param_groups[0].get("lr", args.lr)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.3f} | val_loss={val_loss:.3f}"
                f" | lr={float(lr_now):.6g} | improved={str(improved).lower()}{metrics_str}"
            )
            print(f"DT_PROGRESS_EPOCH {epoch}", flush=True)

            if bool(args.early_stopping) and epochs_without_improvement >= int(args.patience):
                stopped_early = True
                print(
                    f"EARLY_STOPPING triggered at epoch={epoch} best_epoch={best_epoch} "
                    f"best_val_loss={float(best_val_loss):.6f}",
                    flush=True,
                )
                break

        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=600, bbox_inches="tight")
        plt.close()

        if not bool(args.save_best_only):
            torch.save(model.state_dict(), model_path)
        elif not os.path.exists(model_path):
            torch.save(model.state_dict(), model_path)

        if bool(args.save_best_only) and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"--train_model=false but no trained model found at '{model_path}'.")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    _, _, val_y_hat_all, val_graphs_list, val_labels_all = eval_epoch(
        model=model,
        task=task,
        loader=val_loader,
        device=device,
        max_neighbors=args.max_neighbors,
        neighbor_sampling=args.neighbor_sampling,
        num_neighbors=args.num_neighbors,
        seed_count=args.seed_count,
        seed_strategy=args.seed_strategy,
        max_sub_batches=args.max_sub_batches,
        seed_batch_size=args.seed_batch_size,
        task_name=task.name,
        aux_policy_loss=bool(args.aux_policy_loss),
        aux_policy_loss_weight=float(args.aux_policy_loss_weight),
        aux_policy_target_name=str(args.aux_policy_target_name),
        pairwise_policy_ranking_loss=bool(args.pairwise_policy_ranking_loss),
        pairwise_policy_ranking_weight=float(args.pairwise_policy_ranking_weight),
        pairwise_policy_margin=float(args.pairwise_policy_margin),
        pairwise_policy_min_target_gap=float(args.pairwise_policy_min_target_gap),
        oracle_best_action_loss=bool(args.oracle_best_action_loss),
        oracle_best_action_loss_weight=float(args.oracle_best_action_loss_weight),
    )

    test_y_hat_all = None
    test_graphs_list = None
    test_labels_all = None

    if test_loader is not None:
        _, _, test_y_hat_all, test_graphs_list, test_labels_all = eval_epoch(
            model=model,
            task=task,
            loader=test_loader,
            device=device,
            max_neighbors=args.max_neighbors,
            neighbor_sampling=args.neighbor_sampling,
            num_neighbors=args.num_neighbors,
            seed_count=args.seed_count,
            seed_strategy=args.seed_strategy,
            max_sub_batches=args.max_sub_batches,
            seed_batch_size=args.seed_batch_size,
            task_name=task.name,
            aux_policy_loss=bool(args.aux_policy_loss),
            aux_policy_loss_weight=float(args.aux_policy_loss_weight),
            aux_policy_target_name=str(args.aux_policy_target_name),
            pairwise_policy_ranking_loss=bool(args.pairwise_policy_ranking_loss),
            pairwise_policy_ranking_weight=float(args.pairwise_policy_ranking_weight),
            pairwise_policy_margin=float(args.pairwise_policy_margin),
            pairwise_policy_min_target_gap=float(args.pairwise_policy_min_target_gap),
            oracle_best_action_loss=bool(args.oracle_best_action_loss),
            oracle_best_action_loss_weight=float(args.oracle_best_action_loss_weight),
            progress_prefix="TEST",
        )

    run_summary: Dict[str, Any] = {
        "task": str(task.name),
        "is_classification": bool(task.is_classification),
        "is_grouped_policy_classification": bool(_is_grouped_policy_classification_task(task.name)),
        "output_dir": out_dir,
        "config": {
            "data_folder": args.data_folder,
            "test_folder": chosen_test_folder,
            "T": int(args.T),
            "sliding_step": int(args.sliding_step),
            "hidden": int(args.hidden),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "transformer_layers": int(args.transformer_layers),
            "sage_layers": int(args.sage_layers),
            "use_cls": bool(args.use_cls),
            "batch_size": int(args.batch_size),
            "cpu_threads": int(args.cpu_threads),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "max_neighbors": int(args.max_neighbors),
            "neighbor_sampling": bool(args.neighbor_sampling),
            "num_neighbors": list(args.num_neighbors),
            "seed_count": int(args.seed_count),
            "seed_strategy": str(args.seed_strategy),
            "seed_batch_size": int(args.seed_batch_size),
            "max_sub_batches": int(args.max_sub_batches),
            "split_seed": int(args.split_seed),
            "train_model": bool(args.train_model),
            "emit_translational_figures": bool(args.emit_translational_figures),
            "fullgraph_attribution_pass": bool(args.fullgraph_attribution_pass),
            "translational_top_k": int(args.translational_top_k),
            "use_action_conditioning": bool(args.use_action_conditioning),
            "action_hidden_dim": int(args.action_hidden_dim),
            "action_interaction_hidden_dim": int(args.action_interaction_hidden_dim),
            "action_interaction_dropout": float(args.action_interaction_dropout),
            "action_feature_dim": int(action_feature_dim),
            "state_summary_feature_dim": int(state_summary_feature_dim),
            "aux_policy_loss": bool(args.aux_policy_loss),
            "aux_policy_loss_weight": float(args.aux_policy_loss_weight),
            "aux_policy_target_name": str(args.aux_policy_target_name),
            "pairwise_policy_ranking_loss": bool(args.pairwise_policy_ranking_loss),
            "pairwise_policy_ranking_weight": float(args.pairwise_policy_ranking_weight),
            "pairwise_policy_margin": float(args.pairwise_policy_margin),
            "pairwise_policy_min_target_gap": float(args.pairwise_policy_min_target_gap),
            "oracle_best_action_loss": bool(args.oracle_best_action_loss),
            "oracle_best_action_loss_weight": float(args.oracle_best_action_loss_weight),
            "require_complete_action_set": bool(args.require_complete_action_set),
            "oracle_best_action_loss_mode": "statewise_softmax_ce",
            "grouped_policy_classification_task": bool(_is_grouped_policy_classification_task(task.name)),
            "base_task_loss_in_policy_mode": False,
            "early_stopping": bool(args.early_stopping),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "save_best_only": bool(args.save_best_only),
            "lr_scheduler_on_plateau": bool(args.lr_scheduler_on_plateau),
            "lr_scheduler_factor": float(args.lr_scheduler_factor),
            "lr_scheduler_patience": int(args.lr_scheduler_patience),
            "lr_scheduler_min_lr": float(args.lr_scheduler_min_lr),
        },
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "training": {
            "best_val_loss": None if best_val_loss is None else float(best_val_loss),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "epochs_completed": int(epochs_completed),
            "stopped_early": bool(stopped_early),
        },
    }
    if getattr(dataset, "use_policy_manifest", False):
        run_summary["policy_manifest_filtering"] = policy_split_filter_summary

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    if _is_grouped_policy_classification_task(task.name):
        def _plot_policy_metric_panel(metrics_dict: Dict[str, float], out_path: str, title: str):
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.axis("off")
            lines = [title]
            for key in sorted(metrics_dict.keys()):
                val = metrics_dict[key]
                try:
                    lines.append(f"{key}: {float(val):.4f}")
                except Exception:
                    lines.append(f"{key}: {val}")
            ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)
            fig.tight_layout()
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)

        val_group_metrics = task.compute_eval_metrics(val_y_hat_all, val_graphs_list, val_labels_all)
        run_summary["validation"] = dict(val_group_metrics)
        _plot_policy_metric_panel(val_group_metrics, os.path.join(out_dir, "confusion_matrix.png"), "Validation policy metrics")
        _plot_policy_metric_panel(val_group_metrics, os.path.join(out_dir, "roc_curve.png"), "Validation policy metrics")

        if (test_loader is not None) and (test_y_hat_all is not None) and test_y_hat_all.size(0) > 0:
            test_group_metrics = task.compute_eval_metrics(test_y_hat_all, test_graphs_list, test_labels_all)
            run_summary["test"] = dict(test_group_metrics)
            _plot_policy_metric_panel(test_group_metrics, os.path.join(out_dir, "confusion_matrix_test.png"), "Test policy metrics")
            _plot_policy_metric_panel(test_group_metrics, os.path.join(out_dir, "roc_curve_test.png"), "Test policy metrics")
        elif test_loader is not None:
            print("⚠️ Test dataset produced 0 windows for this T/sliding_step; skipping test plots.")
    elif task.is_classification:
        from sklearn.metrics import confusion_matrix

        y_true_val = task.get_targets(val_graphs_list, val_labels_all).cpu().numpy().astype(int).reshape(-1)
        probs_val = F.softmax(val_y_hat_all, dim=1).cpu().numpy()
        preds_val = probs_val.argmax(axis=1)

        class_names = _get_class_names(task, task.out_dim)

        cm_val = confusion_matrix(y_true_val, preds_val, labels=list(range(task.out_dim)))
        plot_confusion_matrix_annotated(
            cm=cm_val,
            out_path=os.path.join(out_dir, "confusion_matrix.png"),
            title="Confusion Matrix (Validation)",
            class_names=class_names,
        )

        plot_roc_curves(
            y_true=y_true_val,
            probs=probs_val,
            n_classes=task.out_dim,
            out_path=os.path.join(out_dir, "roc_curve.png"),
            title="ROC Curve (Validation)",
            class_names=class_names,
        )

        run_summary["validation"] = _classification_split_summary(
            task=task,
            y_true=y_true_val,
            probs=probs_val,
            preds=preds_val,
            class_names=class_names,
        )

        if (test_loader is not None) and (test_y_hat_all is not None) and test_y_hat_all.size(0) > 0:
            y_true_test = task.get_targets(test_graphs_list, test_labels_all).cpu().numpy().astype(int).reshape(-1)
            probs_test = F.softmax(test_y_hat_all, dim=1).cpu().numpy()
            preds_test = probs_test.argmax(axis=1)

            cm_test = confusion_matrix(y_true_test, preds_test, labels=list(range(task.out_dim)))
            plot_confusion_matrix_annotated(
                cm=cm_test,
                out_path=os.path.join(out_dir, "confusion_matrix_test.png"),
                title="Confusion Matrix (Test)",
                class_names=class_names,
            )

            plot_roc_curves(
                y_true=y_true_test,
                probs=probs_test,
                n_classes=task.out_dim,
                out_path=os.path.join(out_dir, "roc_curve_test.png"),
                title="ROC Curve (Test)",
                class_names=class_names,
            )

            run_summary["test"] = _classification_split_summary(
                task=task,
                y_true=y_true_test,
                probs=probs_test,
                preds=preds_test,
                class_names=class_names,
            )
        elif test_loader is not None:
            print("⚠️ Test dataset produced 0 windows for this T/sliding_step; skipping test plots.")
    else:
        y_true_val = task.get_targets(val_graphs_list, val_labels_all).cpu().numpy().reshape(-1)
        y_pred_val = val_y_hat_all.cpu().numpy().reshape(-1)

        def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            plt.figure(figsize=(6, 5))
            plt.scatter(y_true, y_pred, s=12, alpha=0.7)
            if y_true.size > 0:
                lo = float(min(y_true.min(), y_pred.min()))
                hi = float(max(y_true.max(), y_pred.max()))
                plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close()

        def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            resid = y_pred - y_true
            plt.figure(figsize=(6, 5))
            plt.hist(resid, bins=30)
            plt.xlabel("Residual (pred - true)")
            plt.ylabel("Count")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close()

        plot_regression_scatter(
            y_true=y_true_val,
            y_pred=y_pred_val,
            out_path=os.path.join(out_dir, "confusion_matrix.png"),
            title="Predicted vs True (Validation)",
        )
        plot_residual_hist(
            y_true=y_true_val,
            y_pred=y_pred_val,
            out_path=os.path.join(out_dir, "roc_curve.png"),
            title="Residuals (Validation)",
        )

        run_summary["validation"] = _regression_split_summary(
            y_true=y_true_val,
            y_pred=y_pred_val,
        )

        if (test_loader is not None) and (test_y_hat_all is not None) and test_y_hat_all.size(0) > 0:
            y_true_test = task.get_targets(test_graphs_list, test_labels_all).cpu().numpy().reshape(-1)
            y_pred_test = test_y_hat_all.cpu().numpy().reshape(-1)

            plot_regression_scatter(
                y_true=y_true_test,
                y_pred=y_pred_test,
                out_path=os.path.join(out_dir, "confusion_matrix_test.png"),
                title="Predicted vs True (Test)",
            )
            plot_residual_hist(
                y_true=y_true_test,
                y_pred=y_pred_test,
                out_path=os.path.join(out_dir, "roc_curve_test.png"),
                title="Residuals (Test)",
            )

            run_summary["test"] = _regression_split_summary(
                y_true=y_true_test,
                y_pred=y_pred_test,
            )
        else:
            plt.figure(figsize=(6, 5))
            plt.axis("off")
            plt.text(0.5, 0.5, "No test results available", ha="center", va="center")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "confusion_matrix_test.png"), dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(out_dir, "roc_curve_test.png"), dpi=600, bbox_inches="tight")
            plt.close()

    _write_run_summary_files(out_dir=out_dir, summary=run_summary)

    # -------------------------------------------------------------------------
    # RESTORED: Attention heatmaps (node attention vs performance)
    # -------------------------------------------------------------------------
    try:
        train_node_vocab_inv = _load_node_vocab_inv(args.data_folder)
    except Exception:
        train_node_vocab_inv = None

    test_node_vocab_inv = None
    if chosen_test_folder is not None:
        try:
            test_node_vocab_inv = _load_node_vocab_inv(chosen_test_folder)
        except Exception:
            test_node_vocab_inv = None

    heatmap_neighbor_sampling = bool(args.neighbor_sampling)
    heatmap_max_neighbors = int(args.max_neighbors)
    if bool(args.neighbor_sampling) and bool(args.fullgraph_attribution_pass):
        heatmap_neighbor_sampling = False
        heatmap_max_neighbors = -1

    save_attention_heatmap(
        model=model,
        task=task,
        loader=val_loader,
        device=device,
        max_neighbors=heatmap_max_neighbors,
        neighbor_sampling=heatmap_neighbor_sampling,
        num_neighbors=args.num_neighbors,
        seed_count=args.seed_count,
        seed_strategy=args.seed_strategy,
        max_sub_batches=args.max_sub_batches,
        seed_batch_size=args.seed_batch_size,
        out_png=os.path.join(out_dir, "attention_heatmap.png"),
        out_csv=os.path.join(out_dir, "attention_heatmap.csv"),
        split_name="Validation",
        node_vocab_inv=train_node_vocab_inv,
        top_k=args.attn_top_k,
        rank_by=args.attn_rank_by,
    )

    if test_loader is not None:
        save_attention_heatmap(
            model=model,
            task=task,
            loader=test_loader,
            device=device,
            max_neighbors=heatmap_max_neighbors,
            neighbor_sampling=heatmap_neighbor_sampling,
            num_neighbors=args.num_neighbors,
            seed_count=args.seed_count,
            seed_strategy=args.seed_strategy,
            max_sub_batches=args.max_sub_batches,
            seed_batch_size=args.seed_batch_size,
            out_png=os.path.join(out_dir, "attention_heatmap_test.png"),
            out_csv=os.path.join(out_dir, "attention_heatmap_test.csv"),
            split_name="Test",
            node_vocab_inv=test_node_vocab_inv,
            top_k=args.attn_top_k,
            rank_by=args.attn_rank_by,
        )

    if bool(args.emit_translational_figures):
        export_translational_figures_fullgraph(
            model=model,
            task=task,
            loader=val_loader,
            device=device,
            out_dir=out_dir,
            split_name="Validation",
            translational_top_k=args.translational_top_k,
            trained_with_neighbor_sampling=bool(args.neighbor_sampling),
            fullgraph_attribution_pass=bool(args.fullgraph_attribution_pass),
        )
        if test_loader is not None:
            export_translational_figures_fullgraph(
                model=model,
                task=task,
                loader=test_loader,
                device=device,
                out_dir=out_dir,
                split_name="Test",
                translational_top_k=args.translational_top_k,
                trained_with_neighbor_sampling=bool(args.neighbor_sampling),
                fullgraph_attribution_pass=bool(args.fullgraph_attribution_pass),
            )


if __name__ == "__main__":
    main()
