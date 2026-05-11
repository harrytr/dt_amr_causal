#!/usr/bin/env python3
"""
evaluate_policy_selector.py
==========================

Evaluate an intervention-conditioned AMR policy selector on a policy-manifest
causal dataset.

Purpose
-------
Given a trained action-conditioned AMRDyGFormer and a policy-manifest dataset
produced by build_causal_policy_dataset.py, this script:

1. Scores every (state, action) row in the chosen split.
2. Chooses the predicted best action per state.
3. Compares it with the simulator-oracle best action using manifest targets.
4. Writes per-action and per-state tables plus summary JSON/TXT.

This is the decision-evaluation step that should come *before* adding a MILP
layer. It validates whether the model ranks available actions well on held-out
states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import textwrap


def _adaptive_action_plot_params(n_actions: int, panel: bool = False) -> Dict[str, Any]:
    """
    Return plotting parameters that keep action-label heavy plots readable as
    the intervention menu grows.

    The values intentionally scale sub-linearly after moderate action counts:
    figures become larger, but font sizes and annotation density are reduced so
    dashboards remain usable with many interventions.
    """
    n = max(1, int(n_actions))

    if panel:
        return {
            "tick_fontsize": max(4.6, min(8.5, 9.0 - 0.18 * max(0, n - 8))),
            "annot_fontsize": max(4.2, min(8.0, 8.5 - 0.24 * max(0, n - 8))),
            "title_fontsize": 11.0 if n <= 14 else 10.2,
            "axis_fontsize": 9.0 if n <= 14 else 8.2,
            "cell_text_mode": "full" if n <= 10 else ("compact" if n <= 18 else "sparse"),
            "wrap_width": 18 if n <= 10 else (14 if n <= 18 else 11),
            "x_rotation": 35 if n <= 10 else (45 if n <= 18 else 55),
            "grid_linewidth": 0.85 if n <= 14 else 0.55,
            "cbar_fraction": 0.040 if n <= 14 else 0.030,
            "cbar_pad": 0.020 if n <= 14 else 0.012,
            "note_fontsize": max(4.8, min(8.0, 8.2 - 0.16 * max(0, n - 8))),
        }

    return {
        "tick_fontsize": max(5.4, min(10.0, 10.5 - 0.15 * max(0, n - 8))),
        "annot_fontsize": max(4.8, min(10.0, 10.0 - 0.22 * max(0, n - 8))),
        "title_fontsize": 12.5 if n <= 18 else 11.5,
        "axis_fontsize": 11.0 if n <= 18 else 10.0,
        "cell_text_mode": "full" if n <= 14 else ("compact" if n <= 24 else "sparse"),
        "wrap_width": 24 if n <= 10 else (18 if n <= 18 else 14),
        "x_rotation": 30 if n <= 10 else (45 if n <= 20 else 60),
        "grid_linewidth": 1.0 if n <= 18 else 0.6,
        "cbar_fraction": 0.043 if n <= 18 else 0.032,
        "cbar_pad": 0.030 if n <= 18 else 0.018,
        "note_fontsize": max(5.5, min(8.5, 8.8 - 0.12 * max(0, n - 10))),
    }


def _wrap_action_labels(labels: Sequence[str], width: int) -> List[str]:
    """Wrap long intervention labels without changing their underlying names."""
    out: List[str] = []
    width = max(8, int(width))
    for label in labels:
        text = str(label).replace("__", "\n")
        wrapped_parts: List[str] = []
        for part in text.split("\n"):
            part = part.strip()
            if part == "":
                continue
            wrapped_parts.append(textwrap.fill(part, width=width, break_long_words=False, break_on_hyphens=True))
        out.append("\n".join(wrapped_parts) if wrapped_parts else str(label))
    return out


def _format_confusion_cell_text(count: float, pct: float, is_tie_diag: bool, mode: str) -> str:
    """Reduce annotation density as confusion matrices become large."""
    star = "*" if is_tie_diag else ""
    if str(mode) == "sparse":
        if count <= 0:
            return ""
        return f"{count:.0f}{star}"
    if str(mode) == "compact":
        if count <= 0 and pct <= 0:
            return ""
        return f"{count:.0f}{star}\n{pct:.0f}%"
    return f"{count:.0f}{star}\n{pct:.1f}%"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from temporal_graph_dataset import TemporalGraphDataset, collate_temporal_graph_batch
from models_amr import AMRDyGFormer
from tasks import TASK_REGISTRY, BaseTask, get_task

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _nature_axes(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', labelsize=10, width=0.8, length=3)
    ax.grid(False)


def _savefig600(path: str):
    plt.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')


def _format_action_param_value(x: Any) -> str:
    if isinstance(x, bool):
        return "1" if x else "0"
    try:
        fx = float(x)
        if float(fx).is_integer():
            return str(int(fx))
        return f"{fx:.3g}"
    except Exception:
        return str(x)


def _coerce_action_params_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    candidate_keys = [
        "action_params",
        "action_params_json",
        "intervention_params",
        "intervention_params_json",
        "params",
        "params_json",
        "action_json",
        "intervention_json",
    ]

    for key in candidate_keys:
        raw = row.get(key, None)
        if raw is None or raw == "":
            continue

        if isinstance(raw, dict):
            return dict(raw)

        if isinstance(raw, str):
            s = raw.strip()
            if s == "":
                continue
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    if "params" in parsed and isinstance(parsed["params"], dict):
                        return dict(parsed["params"])
                    return parsed
            except Exception:
                pass

    # Fallback: recover common parameter columns directly if manifest wrote them flat
    flat_keys = [
        "frequency_days",
        "screen_on_admission",
        "delay_days",
        "screen_result_delay_days",
        "isolation_mult",
        "isolation_days",
    ]
    out: Dict[str, Any] = {}
    for key in flat_keys:
        if key in row and row.get(key, "") != "":
            out[key] = row.get(key)
    return out


def _build_candidate_action_display_map(candidate_interventions_json: str) -> Dict[str, str]:
    path = str(candidate_interventions_json or "").strip()
    if path == "":
        return {"baseline": "baseline"}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("candidate_interventions_json must contain a JSON object or a list of objects.")

    action_display_map: Dict[str, str] = {"baseline": "baseline"}
    seen_ids = {"baseline"}

    for idx, raw_item in enumerate(payload, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError("Each candidate intervention must be a JSON object.")

        spec_name = str(raw_item.get("name", "")).strip()
        target_type = str(raw_item.get("target_type", "")).strip()
        target_id = str(raw_item.get("target_id", "")).strip()

        suffix = spec_name.lower()
        if target_type:
            suffix += f"__{target_type.lower()}"
        if target_id:
            safe_target = re.sub(r"[^A-Za-z0-9._-]+", "-", target_id)
            suffix += f"__{safe_target}"

        aid = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-") or f"action_{idx:02d}"
        base_aid = aid
        bump = 2
        while aid in seen_ids:
            aid = f"{base_aid}_{bump}"
            bump += 1
        seen_ids.add(aid)

        display_name = str(raw_item.get("display_name", raw_item.get("label", ""))).strip()
        if display_name == "":
            display_name = spec_name if spec_name else aid
        action_display_map[aid] = display_name

    return action_display_map


def _make_action_display_name(
    row: Dict[str, Any],
    candidate_label_map: Optional[Dict[str, str]] = None,
) -> str:
    if int(_safe_int(row.get("is_baseline", 0), 0)) == 1:
        return "baseline"

    action_id = str(row.get("action_id", "")).strip()
    if candidate_label_map is not None:
        mapped = str(candidate_label_map.get(action_id, "")).strip()
        if mapped != "":
            return mapped

    base_name = str(row.get("action_name", "")).strip()
    if base_name == "":
        base_name = str(row.get("action_id", "")).strip()

    params = _coerce_action_params_dict(row)

    if not params:
        return base_name if base_name else str(row.get("action_id", ""))

    preferred_order = [
        "frequency_days",
        "screen_on_admission",
        "delay_days",
        "screen_result_delay_days",
        "isolation_mult",
        "isolation_days",
    ]

    items: List[str] = []
    seen = set()

    for key in preferred_order:
        if key in params:
            items.append(f"{key}={_format_action_param_value(params[key])}")
            seen.add(key)

    for key in sorted(params.keys()):
        if key in seen:
            continue
        items.append(f"{key}={_format_action_param_value(params[key])}")

    return f"{base_name}({', '.join(items)})"


def _safe_label_map(action_ids: Sequence[str], per_rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for aid in action_ids:
        aid_s = str(aid)
        for r in per_rows:
            if str(r.get("action_id", "")) == aid_s:
                nm = str(r.get("action_display_name", "")).strip()
                if nm:
                    out[aid_s] = nm
                    break

            if str(r.get("predicted_best_action_id", "")) == aid_s:
                nm = str(r.get("predicted_best_action_display_name", "")).strip()
                if nm:
                    out[aid_s] = nm
                    break

            if str(r.get("oracle_best_action_id", "")) == aid_s:
                nm = str(r.get("oracle_best_action_display_name", "")).strip()
                if nm:
                    out[aid_s] = nm
                    break

        if aid_s not in out:
            out[aid_s] = aid_s
    return out

def _plot_action_selection_confusion(
    per_state_rows: Sequence[Dict[str, Any]],
    out_path: str,
    all_action_ids: Optional[Sequence[str]] = None,
    action_label_map: Optional[Mapping[str, str]] = None,
) -> None:
    derived_action_ids = sorted({str(r['predicted_best_action_id']) for r in per_state_rows} | {str(r['oracle_best_action_id']) for r in per_state_rows})
    action_ids = [str(a).strip() for a in (all_action_ids or derived_action_ids) if str(a).strip() != ""]
    if not action_ids:
        action_ids = derived_action_ids

    label_map = dict(action_label_map or {})
    fallback_label_map = _safe_label_map(action_ids, per_state_rows)
    for aid, label in fallback_label_map.items():
        label_map.setdefault(str(aid), str(label))
    for aid in action_ids:
        label_map.setdefault(str(aid), str(aid))

    idx = {a: i for i, a in enumerate(action_ids)}
    cm = np.zeros((len(action_ids), len(action_ids)), dtype=float)
    cm_tie_diag = np.zeros((len(action_ids), len(action_ids)), dtype=float)

    for r in per_state_rows:
        oracle_a = str(r["oracle_best_action_id"])
        pred_a = str(r["predicted_best_action_id"])
        tied_ids_raw = r.get("oracle_tied_best_action_ids_json", "")

        try:
            tied_ids = json.loads(tied_ids_raw) if str(tied_ids_raw).strip() != "" else [oracle_a]
        except Exception:
            tied_ids = [oracle_a]

        tied_ids = {str(a) for a in tied_ids}

        if oracle_a not in idx or pred_a not in idx:
            continue

        i = idx[oracle_a]

        if pred_a in tied_ids:
            j = i
            cm[i, j] += 1.0
            if pred_a != oracle_a:
                cm_tie_diag[i, j] += 1.0
        else:
            j = idx[pred_a]
            cm[i, j] += 1.0

    row_sums = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    n_actions = len(action_ids)
    plot_params = _adaptive_action_plot_params(n_actions, panel=False)
    fig_w = max(6.8, min(22.0, 1.05 * n_actions + 4.2))
    fig_h = max(5.8, min(22.0, 0.86 * n_actions + 3.4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cmn, cmap='YlGnBu', vmin=0.0, vmax=1.0, aspect='auto')
    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=float(plot_params["cbar_fraction"]),
        pad=float(plot_params["cbar_pad"]),
    )
    cbar.set_label('Row-normalized proportion', fontsize=float(plot_params["axis_fontsize"]))
    cbar.ax.tick_params(labelsize=max(5.0, float(plot_params["tick_fontsize"]) - 1.0))

    names = _wrap_action_labels([label_map[a] for a in action_ids], width=int(plot_params["wrap_width"]))
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(
        names,
        rotation=float(plot_params["x_rotation"]),
        ha='right',
        fontsize=float(plot_params["tick_fontsize"]),
        rotation_mode='anchor',
    )
    ax.set_yticks(range(n_actions))
    ax.set_yticklabels(names, fontsize=float(plot_params["tick_fontsize"]))
    ax.set_xlabel('Predicted best action', fontsize=float(plot_params["axis_fontsize"]))
    ax.set_ylabel('Oracle best action', fontsize=float(plot_params["axis_fontsize"]))
    ax.set_title(
        'Tie-aware policy selection confusion matrix\n* diagonal includes tie-equivalent predictions',
        fontsize=float(plot_params["title_fontsize"]),
        fontweight='bold',
        pad=12,
    )
    ax.set_xticks(np.arange(-0.5, n_actions, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_actions, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=float(plot_params["grid_linewidth"]))
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(axis='both', which='major', pad=2)

    cell_text_mode = str(plot_params["cell_text_mode"])
    for i in range(n_actions):
        for j in range(n_actions):
            pct = 100.0 * cmn[i, j]
            txt = _format_confusion_cell_text(
                count=float(cm[i, j]),
                pct=float(pct),
                is_tie_diag=bool(i == j and cm_tie_diag[i, j] > 0),
                mode=cell_text_mode,
            )
            if txt == "":
                continue

            color = 'white' if cmn[i, j] >= 0.45 else 'black'
            ax.text(
                j,
                i,
                txt,
                ha='center',
                va='center',
                fontsize=float(plot_params["annot_fontsize"]),
                fontweight='bold',
                color=color,
                linespacing=0.85,
            )

    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)


def _plot_regret_histogram(per_state_rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    regrets = np.asarray([float(r['regret']) for r in per_state_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bins = min(20, max(6, int(np.sqrt(max(1, len(regrets)))) + 2))
    ax.hist(regrets, bins=bins, edgecolor='white', linewidth=0.8)
    mean_r = float(np.mean(regrets)) if regrets.size else float('nan')
    med_r = float(np.median(regrets)) if regrets.size else float('nan')
    if np.isfinite(mean_r):
        ax.axvline(mean_r, linestyle='--', linewidth=1.4, color='black', label=f'Mean = {mean_r:.3f}')
    if np.isfinite(med_r):
        ax.axvline(med_r, linestyle=':', linewidth=1.6, color='dimgray', label=f'Median = {med_r:.3f}')
    ax.set_title('Policy regret distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Regret', fontsize=11)
    ax.set_ylabel('States', fontsize=11)
    _nature_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)


def _plot_baseline_gain(per_state_rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    vals = [float(r['baseline_oracle_delta']) for r in per_state_rows if str(r.get('baseline_oracle_delta', '')) not in {'', 'nan', 'NaN'} and np.isfinite(float(r['baseline_oracle_delta']))]
    arr = np.asarray(vals, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    if arr.size == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No baseline comparison available', ha='center', va='center', fontsize=11)
    else:
        parts = ax.violinplot([arr], positions=[1], widths=0.65, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.30)
            pc.set_edgecolor('none')
        ax.boxplot([arr], positions=[1], widths=0.22, patch_artist=True, boxprops=dict(facecolor='white', linewidth=1.0), medianprops=dict(linewidth=1.4, color='black'), whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
        jitter = 0.06 * (np.random.RandomState(0).rand(arr.size) - 0.5)
        ax.scatter(1 + jitter, arr, s=22, alpha=0.65, edgecolors='white', linewidths=0.4)
        ax.axhline(0.0, color='black', linewidth=0.9, linestyle='--')
        ax.set_xticks([1])
        ax.set_xticklabels(['Selected policy vs baseline'], fontsize=10)
        ax.set_ylabel('Oracle gain over baseline', fontsize=11)
        ax.set_title('Improvement relative to baseline', fontsize=13, fontweight='bold')
        _nature_axes(ax)
    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)

def _plot_action_selection_counts(
    summary: Dict[str, Any],
    per_state_rows: Sequence[Dict[str, Any]],
    out_path: str,
    all_action_ids: Optional[Sequence[str]] = None,
    action_label_map: Optional[Mapping[str, str]] = None,
) -> None:
    summary_catalog = summary.get("action_catalog", []) if isinstance(summary, dict) else []
    summary_action_ids = [
        str(entry.get("action_id", "")).strip()
        for entry in summary_catalog
        if isinstance(entry, dict) and str(entry.get("action_id", "")).strip() != ""
    ]
    derived_action_ids = sorted({
        str(r.get("oracle_best_action_id", ""))
        for r in per_state_rows
    } | {
        str(r.get("predicted_best_action_id", ""))
        for r in per_state_rows
    })

    action_ids = [str(a).strip() for a in (all_action_ids or summary_action_ids or derived_action_ids) if str(a).strip() != ""]
    if not action_ids:
        action_ids = derived_action_ids

    label_map = dict(action_label_map or {})
    fallback_label_map = _safe_label_map(action_ids, per_state_rows)
    for aid, label in fallback_label_map.items():
        label_map.setdefault(str(aid), str(label))
    for aid in action_ids:
        label_map.setdefault(str(aid), str(aid))

    def _counts_for_split(split_name: str):
        oracle_counts = Counter()
        correct_counts = Counter()
        wrong_counts = Counter()

        for row in per_state_rows:
            row_split = str(row.get("split", "")).strip().lower()
            if row_split != split_name:
                continue

            oracle_a = str(row.get("oracle_best_action_id", ""))
            pred_a = str(row.get("predicted_best_action_id", ""))
            tied_ids_raw = row.get("oracle_tied_best_action_ids_json", "")
            try:
                tied_ids = set(json.loads(tied_ids_raw)) if str(tied_ids_raw).strip() != "" else {oracle_a}
            except Exception:
                tied_ids = {oracle_a}
            if oracle_a not in action_ids:
                continue
            oracle_counts[oracle_a] += 1
            if pred_a in tied_ids:
                correct_counts[oracle_a] += 1
            else:
                wrong_counts[oracle_a] += 1

        oracle = np.asarray([int(oracle_counts.get(a, 0)) for a in action_ids], dtype=float)
        correct = np.asarray([int(correct_counts.get(a, 0)) for a in action_ids], dtype=float)
        wrong = np.asarray([int(wrong_counts.get(a, 0)) for a in action_ids], dtype=float)
        return oracle, correct, wrong

    train_oracle, train_correct, train_wrong = _counts_for_split("train")
    test_oracle, test_correct, test_wrong = _counts_for_split("test")

    x = np.arange(len(action_ids))
    w = 0.34
    fig_w = max(8.4, 1.35 * len(action_ids) + 4.4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    # Train stacked bars
    ax.bar(
        x - w / 2,
        train_correct,
        width=w,
        label="Train correct",
        edgecolor="white",
        linewidth=0.7,
        color="#8fd19e",
    )
    ax.bar(
        x - w / 2,
        train_wrong,
        width=w,
        bottom=train_correct,
        label="Train missed",
        edgecolor="white",
        linewidth=0.7,
        color="#ff6b6b",
    )

    # Test stacked bars
    ax.bar(
        x + w / 2,
        test_correct,
        width=w,
        label="Test correct",
        edgecolor="white",
        linewidth=0.7,
        color="#2ca02c",
    )
    ax.bar(
        x + w / 2,
        test_wrong,
        width=w,
        bottom=test_correct,
        label="Test missed",
        edgecolor="white",
        linewidth=0.7,
        color="#d62728",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[a] for a in action_ids], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("States", fontsize=11)
    ax.set_title("Oracle states split into correct vs missed (train vs test)", fontsize=13, fontweight="bold")
    _nature_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)


def _plot_predicted_vs_oracle_scores(per_row_records: Sequence[Dict[str, Any]], out_path: str) -> None:
    xs = np.asarray([float(r['pred_score']) for r in per_row_records], dtype=float)
    ys = np.asarray([float(r['oracle_value']) for r in per_row_records], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if xs.size == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No finite scores available', ha='center', va='center', fontsize=11)
    else:
        ax.scatter(xs, ys, s=18, alpha=0.65, edgecolors='white', linewidths=0.3)
        if xs.size >= 2 and (not np.allclose(np.std(xs), 0.0)) and (not np.allclose(np.std(ys), 0.0)):
            coef = np.polyfit(xs, ys, 1)
            xx = np.linspace(float(xs.min()), float(xs.max()), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, linewidth=1.4, color='black')
            corr = float(np.corrcoef(xs, ys)[0, 1])
            ax.text(0.03, 0.97, f'Pearson r = {corr:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=9.5)
        ax.set_xlabel('Predicted action score', fontsize=11)
        ax.set_ylabel('Oracle action value', fontsize=11)
        ax.set_title('Predicted score vs oracle value', fontsize=13, fontweight='bold')
        _nature_axes(ax)
    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)


def _plot_policy_eval_dashboard(
    summary: Dict[str, Any],
    panel_a_state_rows: Sequence[Dict[str, Any]],
    metric_state_rows: Sequence[Dict[str, Any]],
    metric_per_row_records: Sequence[Dict[str, Any]],
    per_state_rows: Sequence[Dict[str, Any]],
    out_path: str,
    all_action_ids: Optional[Sequence[str]] = None,
    action_label_map: Optional[Mapping[str, str]] = None,
) -> None:
    dashboard_action_ids_for_layout = [
        str(a).strip()
        for a in (all_action_ids or [])
        if str(a).strip() != ""
    ]
    if not dashboard_action_ids_for_layout:
        dashboard_action_ids_for_layout = sorted({
            str(r.get("oracle_best_action_id", ""))
            for r in metric_state_rows
            if str(r.get("oracle_best_action_id", "")).strip() != ""
        } | {
            str(r.get("predicted_best_action_id", ""))
            for r in metric_state_rows
            if str(r.get("predicted_best_action_id", "")).strip() != ""
        })

    dashboard_n_actions = max(1, len(dashboard_action_ids_for_layout))
    large_action_menu = dashboard_n_actions > 10
    fig_w = max(16.0, min(24.0, 16.0 + 0.38 * max(0, dashboard_n_actions - 8)))
    fig_h = max(10.0, min(18.0, 10.0 + 0.34 * max(0, dashboard_n_actions - 8)))
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    if large_action_menu:
        gs = fig.add_gridspec(
            3,
            3,
            width_ratios=[1.35, 1.35, 1.0],
            height_ratios=[0.95, 1.45, 0.85],
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0:2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[2, :])
    else:
        gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], height_ratios=[1.0, 1.0])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        # A: train/test oracle states split into correct vs missed predictions
    derived_panel_action_ids = sorted({
        str(r.get("oracle_best_action_id", ""))
        for r in panel_a_state_rows
    } | {
        str(r.get("predicted_best_action_id", ""))
        for r in panel_a_state_rows
    })
    action_ids = [str(a).strip() for a in (all_action_ids or derived_panel_action_ids) if str(a).strip() != ""]
    if not action_ids:
        action_ids = derived_panel_action_ids

    label_map = dict(action_label_map or {})
    fallback_panel_label_map = _safe_label_map(action_ids, panel_a_state_rows)
    for aid, label in fallback_panel_label_map.items():
        label_map.setdefault(str(aid), str(label))
    for aid in action_ids:
        label_map.setdefault(str(aid), str(aid))

    def _counts_for_split(split_name: str):
        oracle_counts = Counter()
        correct_counts = Counter()
        wrong_counts = Counter()

        for row in panel_a_state_rows:
            row_split = str(row.get("split", "")).strip().lower()
            if row_split != split_name:
                continue

            oracle_a = str(row.get("oracle_best_action_id", ""))
            pred_a = str(row.get("predicted_best_action_id", ""))
            tied_ids_raw = row.get("oracle_tied_best_action_ids_json", "")
            try:
                tied_ids = set(json.loads(tied_ids_raw)) if str(tied_ids_raw).strip() != "" else {oracle_a}
            except Exception:
                tied_ids = {oracle_a}
            oracle_counts[oracle_a] += 1
            if pred_a in tied_ids:
                correct_counts[oracle_a] += 1
            else:
                wrong_counts[oracle_a] += 1

        oracle = np.asarray([int(oracle_counts.get(a, 0)) for a in action_ids], dtype=float)
        correct = np.asarray([int(correct_counts.get(a, 0)) for a in action_ids], dtype=float)
        wrong = np.asarray([int(wrong_counts.get(a, 0)) for a in action_ids], dtype=float)
        return oracle, correct, wrong

    train_oracle, train_correct, train_wrong = _counts_for_split("train")
    test_oracle, test_correct, test_wrong = _counts_for_split("test")

    x = np.arange(len(action_ids))
    w = 0.34

    ax1.bar(
        x - w / 2,
        train_correct,
        width=w,
        label="Train correct",
        edgecolor="white",
        linewidth=0.7,
        color="#8fd19e",
    )
    ax1.bar(
        x - w / 2,
        train_wrong,
        width=w,
        bottom=train_correct,
        label="Train missed",
        edgecolor="white",
        linewidth=0.7,
        color="#ff6b6b",
    )

    ax1.bar(
        x + w / 2,
        test_correct,
        width=w,
        label="Test correct",
        edgecolor="white",
        linewidth=0.7,
        color="#2ca02c",
    )
    ax1.bar(
        x + w / 2,
        test_wrong,
        width=w,
        bottom=test_correct,
        label="Test missed",
        edgecolor="white",
        linewidth=0.7,
        color="#d62728",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map[a] for a in action_ids], rotation=22, ha="right", fontsize=9)
    ax1.set_ylabel("States", fontsize=10)
    ax1.set_title("A  Oracle states: train vs test, correct vs missed", loc="left", fontsize=12.5, fontweight="bold")
    _nature_axes(ax1)
    ax1.legend(frameon=False, fontsize=8.3, ncol=2)

    # B: regret hist
    regrets = np.asarray([float(r['regret']) for r in metric_state_rows], dtype=float)
    bins = min(20, max(6, int(np.sqrt(max(1, len(regrets)))) + 2))
    ax2.hist(regrets, bins=bins, edgecolor='white', linewidth=0.8)
    ax2.axvline(float(np.mean(regrets)), linestyle='--', linewidth=1.3, color='black')
    ax2.axvline(float(np.median(regrets)), linestyle=':', linewidth=1.5, color='dimgray')
    ax2.set_xlabel('Regret', fontsize=10); ax2.set_ylabel('States', fontsize=10)
    ax2.set_title('B  Policy regret distribution (Test)', loc='left', fontsize=12.5, fontweight='bold')
    _nature_axes(ax2)

    # C: baseline gain
    vals = [float(r['baseline_oracle_delta']) for r in metric_state_rows if np.isfinite(float(r['baseline_oracle_delta']))]
    arr = np.asarray(vals, dtype=float)
    if arr.size > 0:
        parts = ax3.violinplot([arr], positions=[1], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.30); pc.set_edgecolor('none')
        ax3.boxplot([arr], positions=[1], widths=0.22, patch_artist=True, boxprops=dict(facecolor='white', linewidth=1.0), medianprops=dict(linewidth=1.4, color='black'), whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
        jitter = 0.06 * (np.random.RandomState(0).rand(arr.size) - 0.5)
        ax3.scatter(1 + jitter, arr, s=18, alpha=0.65, edgecolors='white', linewidths=0.3)
        ax3.axhline(0.0, color='black', linewidth=0.9, linestyle='--')
        ax3.set_xticks([1]); ax3.set_xticklabels(['Selected vs baseline'], fontsize=9)
        ax3.set_ylabel('Oracle gain', fontsize=10)
        _nature_axes(ax3)
    else:
        ax3.axis('off'); ax3.text(0.5,0.5,'No baseline comparison',ha='center',va='center',fontsize=10.5)
    ax3.set_title('C  Improvement relative to baseline', loc='left', fontsize=12.5, fontweight='bold')

    # D: confusion matrix
    derived_cm_action_ids = sorted({str(r['predicted_best_action_id']) for r in metric_state_rows} | {str(r['oracle_best_action_id']) for r in metric_state_rows})
    cm_action_ids = [str(a).strip() for a in (all_action_ids or derived_cm_action_ids) if str(a).strip() != ""]
    if not cm_action_ids:
        cm_action_ids = derived_cm_action_ids
    
    cm_label_map = dict(action_label_map or {})
    fallback_cm_label_map = _safe_label_map(cm_action_ids, metric_state_rows)
    for aid, label in fallback_cm_label_map.items():
        cm_label_map.setdefault(str(aid), str(label))
    for aid in cm_action_ids:
        cm_label_map.setdefault(str(aid), str(aid))
    
    id_to_idx = {a: i for i, a in enumerate(cm_action_ids)}
    cm = np.zeros((len(cm_action_ids), len(cm_action_ids)), dtype=float)
    cm_tie_diag = np.zeros((len(cm_action_ids), len(cm_action_ids)), dtype=float)
    
    for r in metric_state_rows:
        oracle_a = str(r["oracle_best_action_id"])
        pred_a = str(r["predicted_best_action_id"])
        tied_ids_raw = r.get("oracle_tied_best_action_ids_json", "")
    
        try:
            tied_ids = json.loads(tied_ids_raw) if str(tied_ids_raw).strip() != "" else [oracle_a]
        except Exception:
            tied_ids = [oracle_a]
    
        tied_ids = {str(a) for a in tied_ids}
    
        if oracle_a not in id_to_idx or pred_a not in id_to_idx:
            continue
    
        i = id_to_idx[oracle_a]
    
        if pred_a in tied_ids:
            j = i
            cm[i, j] += 1.0
            if pred_a != oracle_a:
                cm_tie_diag[i, j] += 1.0
        else:
            j = id_to_idx[pred_a]
            cm[i, j] += 1.0
    
    rs = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, rs, out=np.zeros_like(cm), where=rs != 0)
    n_cm_actions = len(cm_action_ids)
    panel_d_params = _adaptive_action_plot_params(n_cm_actions, panel=True)
    im = ax4.imshow(cmn, cmap='YlGnBu', vmin=0.0, vmax=1.0, aspect='auto')

    cm_names = _wrap_action_labels(
        [cm_label_map[a] for a in cm_action_ids],
        width=int(panel_d_params["wrap_width"]),
    )
    ax4.set_xticks(range(n_cm_actions))
    ax4.set_xticklabels(
        cm_names,
        rotation=float(panel_d_params["x_rotation"]),
        ha='right',
        fontsize=float(panel_d_params["tick_fontsize"]),
        rotation_mode='anchor',
    )
    ax4.set_yticks(range(n_cm_actions))
    ax4.set_yticklabels(cm_names, fontsize=float(panel_d_params["tick_fontsize"]))
    ax4.set_xlabel('Predicted', fontsize=float(panel_d_params["axis_fontsize"]))
    ax4.set_ylabel('Oracle', fontsize=float(panel_d_params["axis_fontsize"]))
    ax4.set_title(
        'D  Tie-aware confusion matrix (Test)',
        loc='left',
        fontsize=float(panel_d_params["title_fontsize"]),
        fontweight='bold',
    )
    ax4.set_xticks(np.arange(-0.5, n_cm_actions, 1), minor=True)
    ax4.set_yticks(np.arange(-0.5, n_cm_actions, 1), minor=True)
    ax4.grid(which='minor', color='white', linestyle='-', linewidth=float(panel_d_params["grid_linewidth"]))
    ax4.tick_params(which='minor', bottom=False, left=False)
    ax4.tick_params(axis='both', which='major', pad=1.5)

    cell_text_mode = str(panel_d_params["cell_text_mode"])
    for i in range(n_cm_actions):
        for j in range(n_cm_actions):
            pct = 100.0 * cmn[i, j]
            txt = _format_confusion_cell_text(
                count=float(cm[i, j]),
                pct=float(pct),
                is_tie_diag=bool(i == j and cm_tie_diag[i, j] > 0),
                mode=cell_text_mode,
            )
            if txt == "":
                continue

            color = 'white' if cmn[i, j] >= 0.45 else 'black'
            ax4.text(
                j,
                i,
                txt,
                ha='center',
                va='center',
                fontsize=float(panel_d_params["annot_fontsize"]),
                fontweight='bold',
                color=color,
                linespacing=0.82,
            )

    ax4.text(
        0.0,
        -0.18 if n_cm_actions <= 10 else -0.24,
        "* diagonal includes tie-equivalent predictions",
        transform=ax4.transAxes,
        ha="left",
        va="top",
        fontsize=float(panel_d_params["note_fontsize"]),
    )
    cbar = plt.colorbar(
        im,
        ax=ax4,
        fraction=float(panel_d_params["cbar_fraction"]),
        pad=float(panel_d_params["cbar_pad"]),
    )
    cbar.ax.tick_params(labelsize=max(4.8, float(panel_d_params["tick_fontsize"]) - 1.0))

    # E: AUROC for classification tasks, otherwise keep confidence-gap plot
    is_classification = bool(summary.get("is_classification", False))

    if is_classification:
        roc_payload = _classification_row_auc(
            per_row_records=metric_per_row_records,
            per_state_rows=metric_state_rows,
        )
        fpr = np.asarray(roc_payload.get("fpr", [0.0, 1.0]), dtype=float)
        tpr = np.asarray(roc_payload.get("tpr", [0.0, 1.0]), dtype=float)
        auroc = float(roc_payload.get("auroc", float("nan")))

        if fpr.size >= 2 and tpr.size >= 2 and np.isfinite(auroc):
            ax5.plot(fpr, tpr, linewidth=1.6, color="#1f77b4", label=f"AUROC = {auroc:.3f}")
            ax5.fill_between(fpr, 0.0, tpr, color="#1f77b4", alpha=0.25)
            ax5.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="black")
            ax5.set_xlim(0.0, 1.0)
            ax5.set_ylim(0.0, 1.0)
            ax5.set_xlabel("False positive rate", fontsize=10)
            ax5.set_ylabel("True positive rate", fontsize=10)
            ax5.legend(frameon=False, fontsize=8.8, loc="lower right")
            _nature_axes(ax5)
        else:
            ax5.axis("off")
            ax5.text(0.5, 0.5, "AUROC unavailable", ha="center", va="center", fontsize=10.5)

        ax5.set_title("E  Oracle-best row ROC (Test)", loc="left", fontsize=12.5, fontweight="bold")
    else:
        rows_test = [
            r for r in metric_state_rows
            if str(r.get("split", "")).strip().lower() == "test"
        ]

        gaps = np.asarray(
            [
                float(r["predicted_best_pred_score"]) - float(r["oracle_best_pred_score"])
                for r in rows_test
            ],
            dtype=float,
        )
        correct = np.asarray(
            [int(r["policy_match"]) for r in rows_test],
            dtype=int,
        )

        mask = np.isfinite(gaps)
        gaps = gaps[mask]
        correct = correct[mask]

        if gaps.size > 0:
            x = np.arange(1, gaps.size + 1, dtype=float)
            jitter = 0.06 * (np.random.RandomState(0).rand(gaps.size) - 0.5)

            miss_mask = (correct == 0)
            hit_mask = (correct == 1)

            if np.any(hit_mask):
                ax5.scatter(
                    x[hit_mask] + jitter[hit_mask],
                    gaps[hit_mask],
                    s=20,
                    alpha=0.70,
                    edgecolors='white',
                    linewidths=0.25,
                    color='#2ca02c',
                    label='Correct',
                )

            if np.any(miss_mask):
                ax5.scatter(
                    x[miss_mask] + jitter[miss_mask],
                    gaps[miss_mask],
                    s=20,
                    alpha=0.75,
                    edgecolors='white',
                    linewidths=0.25,
                    color='#d62728',
                    label='Missed',
                )

            ax5.axhline(0.0, color='black', linewidth=1.0, linestyle='--')

            mean_gap = float(np.mean(gaps))
            median_gap = float(np.median(gaps))
            ax5.text(
                0.03,
                0.97,
                f"Mean = {mean_gap:.3f}\nMedian = {median_gap:.3f}",
                transform=ax5.transAxes,
                ha='left',
                va='top',
                fontsize=9,
            )

            ax5.set_xlabel('Test state', fontsize=10)
            ax5.set_ylabel('Pred-best minus oracle-best\n(predictor score)', fontsize=10)
            ax5.legend(frameon=False, fontsize=8.8, loc='best')
            _nature_axes(ax5)
        else:
            ax5.axis('off')
            ax5.text(0.5, 0.5, 'No finite test confidence gaps available', ha='center', va='center', fontsize=10.5)

        ax5.set_title('E  Predictor confidence gap vs oracle-best action (Test)', loc='left', fontsize=12.5, fontweight='bold')
    
    
    # F: headline metrics
    ax6.axis('off')
    classification_metrics = summary.get("classification_metrics", {})
    metrics_text = [
        f"Tie-aware policy accuracy: {float(summary.get('policy_accuracy', float('nan'))):.3f}",
        f"Strict policy accuracy: {float(summary.get('strict_policy_accuracy', float('nan'))):.3f}",
        f"Tie-aware Top-2 accuracy: {float(summary.get('top2_accuracy', float('nan'))):.3f}",
        f"Strict Top-2 accuracy: {float(summary.get('strict_top2_accuracy', float('nan'))):.3f}",
        f"Tie-aware mean regret: {float(summary.get('regret', {}).get('mean', float('nan'))):.4f}",
        f"Tie-aware median regret: {float(summary.get('regret', {}).get('median', float('nan'))):.4f}",
        f"Strict baseline improvement (>0): {float(summary.get('baseline_improvement_rate', float('nan'))):.3f}",
        f"Tie-aware baseline non-worse (>=0): {float(summary.get('baseline_non_worse_rate', float('nan'))):.3f}",
        f"Baseline tie rate (=0): {float(summary.get('baseline_tie_rate', float('nan'))):.3f}",
        f"Tie-state rate: {float(summary.get('tie_state_rate', float('nan'))):.3f}",
    ]

    tie_aware_classification_metrics = summary.get("tie_aware_classification_metrics", {})
    
    if bool(summary.get("is_classification", False)) and isinstance(classification_metrics, dict):
        metrics_text.extend([
            f"Strict macro precision (tie-aware {float(tie_aware_classification_metrics.get('macro_precision', float('nan'))):.3f}): {float(classification_metrics.get('macro_precision', float('nan'))):.3f}",
            f"Strict macro recall (tie-aware {float(tie_aware_classification_metrics.get('macro_recall', float('nan'))):.3f}): {float(classification_metrics.get('macro_recall', float('nan'))):.3f}",
            f"Strict macro F1 (tie-aware {float(tie_aware_classification_metrics.get('macro_f1', float('nan'))):.3f}): {float(classification_metrics.get('macro_f1', float('nan'))):.3f}",
            f"Strict sensitivity (tie-aware {float(tie_aware_classification_metrics.get('macro_sensitivity', float('nan'))):.3f}): {float(classification_metrics.get('macro_sensitivity', float('nan'))):.3f}",
            f"Strict specificity (tie-aware {float(tie_aware_classification_metrics.get('macro_specificity', float('nan'))):.3f}): {float(classification_metrics.get('macro_specificity', float('nan'))):.3f}",
            f"Strict balanced accuracy (tie-aware {float(tie_aware_classification_metrics.get('balanced_accuracy', float('nan'))):.3f}): {float(classification_metrics.get('balanced_accuracy', float('nan'))):.3f}",
            f"AUROC: {float(classification_metrics.get('auroc', float('nan'))):.3f}",
        ])

    metrics_text.extend([
        f"States: {int(summary.get('n_states', 0))}",
        f"Actions: {int(summary.get('n_actions', 0))}",
        f"Oracle metric: {str(summary.get('oracle_metric', ''))}",
    ])
    ax6.text(0.0, 1.0, 'F  Policy evaluation summary', ha='left', va='top', fontsize=12.5, fontweight='bold')
    ax6.text(0.0, 0.86, '\n'.join(metrics_text), ha='left', va='top', fontsize=11, linespacing=1.55)

    fig.suptitle('Policy selector evaluation dashboard', fontsize=16, fontweight='bold', x=0.01, ha='left')
    _savefig600(out_path)
    plt.close(fig)



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


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        try:
            if hasattr(x, "item"):
                return float(x.item())
        except Exception:
            return float(default)
    return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            if hasattr(x, "item"):
                return int(x.item())
        except Exception:
            return int(default)
    return int(default)


def _build_ordered_action_catalog(
    dataset: TemporalGraphDataset,
    per_row_records: Sequence[Dict[str, Any]],
    candidate_label_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    catalog: Dict[str, Dict[str, Any]] = {}

    def _ingest(row: Mapping[str, Any]) -> None:
        aid = str(row.get("action_id", "")).strip()
        if aid == "":
            return
        action_index = _safe_int(row.get("action_index", 10**9), 10**9)
        action_name = str(row.get("action_name", "")).strip() or aid
        display_name = str(row.get("action_display_name", "")).strip()
        if display_name == "":
            try:
                display_name = _make_action_display_name(dict(row), candidate_label_map=candidate_label_map)
            except Exception:
                display_name = action_name
        is_baseline = _safe_int(row.get("is_baseline", 0), 0)

        entry = catalog.get(aid)
        if entry is None:
            catalog[aid] = {
                "action_id": aid,
                "action_name": action_name,
                "action_display_name": display_name or action_name,
                "action_index": int(action_index),
                "is_baseline": int(is_baseline),
            }
            return

        entry["action_index"] = min(int(entry.get("action_index", 10**9)), int(action_index))
        if str(entry.get("action_name", "")).strip() == "" and action_name != "":
            entry["action_name"] = action_name
        if str(entry.get("action_display_name", "")).strip() == "" and display_name != "":
            entry["action_display_name"] = display_name
        if int(is_baseline) == 1:
            entry["is_baseline"] = 1

    for sample in getattr(dataset, "samples", []):
        row_meta = sample.get("row_meta", {}) if isinstance(sample, dict) else {}
        if isinstance(row_meta, Mapping):
            _ingest(row_meta)

    for row in per_row_records:
        if isinstance(row, Mapping):
            _ingest(row)

    ordered_catalog = sorted(
        catalog.values(),
        key=lambda entry: (
            int(entry.get("action_index", 10**9)),
            str(entry.get("action_id", "")),
        ),
    )
    ordered_action_ids = [str(entry["action_id"]) for entry in ordered_catalog]
    action_label_map = {
        str(entry["action_id"]): str(entry.get("action_display_name", entry.get("action_name", entry["action_id"])))
        for entry in ordered_catalog
    }
    return ordered_action_ids, action_label_map, ordered_catalog


def _zero_filled_action_counts(action_ids: Sequence[str], counts: Mapping[str, Any]) -> Dict[str, int]:
    return {str(aid): int(counts.get(str(aid), 0)) for aid in action_ids}


def _zero_filled_display_counts(
    action_ids: Sequence[str],
    action_label_map: Mapping[str, str],
    counts: Mapping[str, Any],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for aid in action_ids:
        aid_s = str(aid)
        label = str(action_label_map.get(aid_s, aid_s))
        out[label] = int(counts.get(aid_s, 0))
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _get_task(task_name: str) -> BaseTask:
    if task_name in TASK_REGISTRY:
        task_obj = TASK_REGISTRY[task_name]
        if isinstance(task_obj, BaseTask):
            return task_obj
        if isinstance(task_obj, type):
            return task_obj()
        if callable(task_obj):
            return task_obj()
        raise TypeError(
            f"TASK_REGISTRY['{task_name}'] must be a BaseTask instance or a callable/class returning one; "
            f"got type={type(task_obj)}"
        )
    return get_task(task_name)


def _infer_feature_dims(dataset: TemporalGraphDataset) -> Tuple[int, int]:
    graphs, _ = dataset[0]
    g = graphs[0]
    in_channels = int(g.x.size(1))
    edge_dim = int(g.edge_attr.size(1)) if hasattr(g, "edge_attr") and g.edge_attr is not None else 0
    return in_channels, edge_dim


def _infer_action_feature_dim(dataset: TemporalGraphDataset) -> int:
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


def _infer_state_summary_feature_dim(dataset: TemporalGraphDataset) -> int:
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


def _resolve_output_activation(task: BaseTask) -> str:
    return str(getattr(task, "output_activation", "identity" if task.is_classification else "softplus")).lower().strip()


def _load_model_from_training_dir(
    dataset: TemporalGraphDataset,
    trained_dir: str,
    task: BaseTask,
    device: torch.device,
    task_name_override: Optional[str],
    use_action_conditioning_override: Optional[bool],
    action_hidden_dim_override: Optional[int],
    action_interaction_hidden_dim_override: Optional[int],
    action_interaction_dropout_override: Optional[float],
) -> Tuple[AMRDyGFormer, Dict[str, Any], str]:
    trained_dir = os.path.abspath(trained_dir)
    metrics_json = os.path.join(trained_dir, "metrics_summary.json")
    model_path = os.path.join(trained_dir, "trained_model.pt")
    if not os.path.isfile(metrics_json):
        raise FileNotFoundError(f"metrics_summary.json not found in trained_dir: {trained_dir}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"trained_model.pt not found in trained_dir: {trained_dir}")

    summary = _load_json(metrics_json)
    cfg = dict(summary.get("config", {}))

    task_name = str(task_name_override or summary.get("task", "")).strip()
    if task_name == "":
        raise ValueError("Unable to resolve task name from trained_dir metrics_summary.json; pass --task explicitly.")

    in_channels, edge_dim = _infer_feature_dims(dataset)
    action_feature_dim = _infer_action_feature_dim(dataset)
    state_summary_feature_dim = _infer_state_summary_feature_dim(dataset)

    use_action_conditioning = bool(
        use_action_conditioning_override
        if use_action_conditioning_override is not None
        else cfg.get("use_action_conditioning", False)
    )
    action_hidden_dim = int(
        action_hidden_dim_override
        if action_hidden_dim_override is not None
        else cfg.get("action_hidden_dim", 32)
    )
    action_interaction_hidden_dim = int(
        action_interaction_hidden_dim_override
        if action_interaction_hidden_dim_override is not None
        else cfg.get("action_interaction_hidden_dim", 128)
    )
    action_interaction_dropout = float(
        action_interaction_dropout_override
        if action_interaction_dropout_override is not None
        else cfg.get("action_interaction_dropout", 0.1)
    )

    model = AMRDyGFormer(
        in_channels=in_channels,
        hidden_channels=int(cfg.get("hidden", 64)),
        edge_dim=edge_dim,
        heads=int(cfg.get("heads", 2)),
        T=int(cfg.get("T", 7)),
        dropout=float(cfg.get("dropout", 0.2)),
        use_cls_token=bool(cfg.get("use_cls", False)),
        n_outputs=int(task.out_dim),
        n_layers=int(cfg.get("transformer_layers", 2)),
        sage_layers=int(cfg.get("sage_layers", 2)),
        use_softplus=(_resolve_output_activation(task) == "softplus"),
        output_activation=_resolve_output_activation(task),
        action_feature_dim=int(action_feature_dim),
        action_hidden_dim=int(action_hidden_dim),
        use_action_conditioning=bool(use_action_conditioning),
        action_interaction_hidden_dim=int(action_interaction_hidden_dim),
        action_interaction_dropout=float(action_interaction_dropout),
        state_summary_feature_dim=int(state_summary_feature_dim),
        state_summary_hidden_dim=int(action_hidden_dim),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model, summary, task_name


def _subset_indices_for_split(dataset: TemporalGraphDataset, split_name: str) -> List[int]:
    if not getattr(dataset, "use_policy_manifest", False):
        raise ValueError("evaluate_policy_selector.py requires a policy-manifest dataset folder.")

    split_norm = str(split_name).strip().lower()
    if split_norm == "val":
        split_norm = "validation"

    idxs: List[int] = []
    for i, sample in enumerate(getattr(dataset, "samples", [])):
        row_meta = sample.get("row_meta", {})
        row_split = str(row_meta.get("split", "")).strip().lower()
        if row_split == "val":
            row_split = "validation"
        if row_split == split_norm:
            idxs.append(i)
    return sorted(idxs)


def _get_row_meta(dataset: TemporalGraphDataset, dataset_index: int) -> Dict[str, Any]:
    return dict(dataset.samples[dataset_index].get("row_meta", {}))


def _task_probability_and_prediction(
    task: BaseTask,
    y_hat: torch.Tensor,
    task_name: str,
    oracle_metric: str,
    oracle_direction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred_score: scalar per sample used for action ranking
      pred_label_or_value: predicted class/value per sample

    For classification tasks, pred_score is an exact task-aware utility that
    matches the trainer's policy utility convention:
      - if the oracle metric is minimized, higher utility = lower burden
      - if the oracle metric is maximized, higher utility = higher gain
    """
    del task_name, oracle_metric

    oracle_direction = str(oracle_direction).strip().lower()

    if task.is_classification:
        if y_hat.dim() != 2:
            raise ValueError(f"Expected classification logits with shape [N, C], got {tuple(y_hat.shape)}")

        if y_hat.size(1) == 1:
            p1_t = torch.sigmoid(y_hat.view(-1))
            pred_label_t = (p1_t >= 0.5).to(torch.int64)
        else:
            probs_t = F.softmax(y_hat, dim=1)
            p1_t = probs_t[:, 1]
            pred_label_t = probs_t.argmax(dim=1).to(torch.int64)

        utility_t = p1_t if oracle_direction == "maximize" else (1.0 - p1_t)
        return (
            utility_t.detach().cpu().numpy().astype(float),
            pred_label_t.detach().cpu().numpy().astype(int),
        )

    pred = y_hat.detach().cpu().numpy()
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred.astype(float), pred.astype(float)


def _auto_oracle_metric(task_name: str) -> Tuple[str, str]:
    """Return (metric_name, direction) where direction is minimize/maximize."""
    m = re.search(r"_h(\d+)$", task_name)
    h = int(m.group(1)) if m else 7
    tn = str(task_name).strip().lower()

    if tn.startswith("endogenous_importation_majority_h") or tn.startswith("endogenous_importation_share_h"):
        return f"y_h{h}_endog_share", "minimize"
    if tn.startswith("endogenous_transmission_majority_h") or tn.startswith("endogenous_transmission_share_h"):
        return f"y_h{h}_trans_share", "minimize"
    if tn.startswith("mechanism_import_share_h"):
        return f"y_h{h}_import_share", "minimize"
    if tn.startswith("mechanism_selection_share_h"):
        return f"y_h{h}_select_share", "minimize"
    if tn.startswith("amr_cr_acq_h"):
        return f"y_h{h}_cr_acq", "minimize"
    if tn.startswith("amr_ir_inf_h"):
        return f"y_h{h}_ir_inf", "minimize"
    if tn.startswith("predict_resistance_emergence_h"):
        return f"y_h{h}_any_res_emergence", "minimize"
    if tn.startswith("transmission_importation_resistant_burden_gain_h"):
        return f"y_h{h}_trans_import_res_gain", "maximize"
    if tn.startswith("oracle_best_action_h"):
        return f"y_h{h}_trans_import_res_gain", "maximize"
    if tn.startswith("transmission_importation_resistant_burden_h"):
        return f"y_h{h}_trans_import_res", "minimize"
    if tn.startswith("transmission_resistant_burden_gain_h"):
        return f"y_h{h}_trans_res_gain", "maximize"
    if tn.startswith("transmission_resistant_burden_h"):
        return f"y_h{h}_trans_res", "minimize"
    if tn.startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac", "minimize"
    if tn.startswith("optimal_screening_target_h"):
        return f"y_h{h}_screening_gain", "maximize"
    if tn.startswith("staff_mediation_effect_h"):
        return f"y_h{h}_transmissions", "minimize"

    # conservative fallback
    return f"y_h{h}_endog_share", "minimize"


def _apply_direction(values: Iterable[float], direction: str) -> List[float]:
    direction = str(direction).strip().lower()
    vals = [float(v) for v in values]
    if direction == "minimize":
        return vals
    if direction == "maximize":
        return [-v for v in vals]
    raise ValueError(f"Unknown direction: {direction}")


def _summarize_regret(values: Sequence[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _normalized_split_name(x: Any) -> str:
    s = str(x).strip().lower()
    return "validation" if s == "val" else s


def _filter_per_state_rows_by_split(
    per_state_rows: Sequence[Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    split_norm = _normalized_split_name(split_name)
    return [
        dict(r)
        for r in per_state_rows
        if _normalized_split_name(r.get("split", "")) == split_norm
    ]


def _filter_per_row_records_by_split(
    per_row_records: Sequence[Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    split_norm = _normalized_split_name(split_name)
    return [
        dict(r)
        for r in per_row_records
        if _normalized_split_name(r.get("split", "")) == split_norm
    ]


def _filter_per_row_records_by_state_ids(
    per_row_records: Sequence[Dict[str, Any]],
    state_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    allowed = {str(s).strip() for s in state_ids if str(s).strip() != ""}
    return [
        dict(r)
        for r in per_row_records
        if str(r.get("state_id", "")).strip() in allowed
    ]


def _state_action_sort_key(row: Mapping[str, Any]) -> Tuple[int, str, str]:
    return (
        _safe_int(row.get("action_index", 10**9), 10**9),
        str(row.get("action_id", "")).strip(),
        str(row.get("action_name", "")).strip(),
    )


def _sorted_state_action_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(r) for r in sorted(rows, key=_state_action_sort_key)]


def _extract_policy_horizon(task_name: str, oracle_metric: str) -> Optional[int]:
    candidates = [str(task_name).strip().lower(), str(oracle_metric).strip().lower()]
    for raw in candidates:
        m = re.search(r"_h(\d+)(?:$|_)", raw)
        if m:
            return int(m.group(1))
        m = re.search(r"y_h(\d+)_", raw)
        if m:
            return int(m.group(1))
    return None


def _stable_best_index(values: Sequence[float]) -> int:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot choose a best element from an empty sequence.")
    return int(np.argsort(arr, kind="mergesort")[0])


def _stable_topk_indices(values: Sequence[float], k: int) -> List[int]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or int(k) <= 0:
        return []
    order = np.argsort(arr, kind="mergesort")
    return [int(x) for x in order[: min(int(k), int(arr.size))].tolist()]


def _oracle_tied_best_indices(rows_sorted: Sequence[Dict[str, Any]], oracle_direction: str) -> List[int]:
    oracle_vals = np.asarray(_apply_direction([r.get("oracle_value", float("nan")) for r in rows_sorted], oracle_direction), dtype=float)
    if oracle_vals.size == 0 or np.any(~np.isfinite(oracle_vals)):
        return []
    best_val = float(np.min(oracle_vals))
    return [
        int(i)
        for i, val in enumerate(oracle_vals.tolist())
        if math.isclose(float(val), best_val, rel_tol=1e-12, abs_tol=1e-12)
    ]


def _oracle_tied_best_action_ids(rows_sorted: Sequence[Dict[str, Any]], oracle_direction: str) -> List[str]:
    return [str(rows_sorted[i].get("action_id", "")) for i in _oracle_tied_best_indices(rows_sorted, oracle_direction)]


def _resolve_state_oracle_row(
    rows: Sequence[Dict[str, Any]],
    task_name: str,
    oracle_metric: str,
    oracle_direction: str,
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
    rows_sorted = _sorted_state_action_rows(rows)
    if len(rows_sorted) == 0:
        return [], None, "empty_state"

    horizon = _extract_policy_horizon(task_name=task_name, oracle_metric=oracle_metric)
    if horizon is not None:
        oracle_index_key = f"oracle_best_action_index_h{horizon}"
        oracle_id_key = f"oracle_best_action_id_h{horizon}"

        stored_index_raw = rows_sorted[0].get(oracle_index_key, "")
        stored_id = str(rows_sorted[0].get(oracle_id_key, "")).strip()
        stored_index: Optional[int] = None
        if str(stored_index_raw).strip() != "":
            stored_index = _safe_int(stored_index_raw, 10**9)

        if stored_index is not None or stored_id != "":
            matches: List[int] = []
            for idx, row in enumerate(rows_sorted):
                match = True
                if stored_index is not None:
                    match = match and (_safe_int(row.get("action_index", 10**9), 10**9) == stored_index)
                if stored_id != "":
                    match = match and (str(row.get("action_id", "")).strip() == stored_id)
                if match:
                    matches.append(idx)

            if len(matches) == 1:
                return rows_sorted, int(matches[0]), None
            if len(matches) > 1:
                return rows_sorted, None, "ambiguous_stored_oracle_action"
            return rows_sorted, None, "missing_stored_oracle_action"

    oracle_vals = np.asarray(_apply_direction([r.get("oracle_value", float("nan")) for r in rows_sorted], oracle_direction), dtype=float)
    if np.any(~np.isfinite(oracle_vals)):
        return rows_sorted, None, "nonfinite_oracle"
    return rows_sorted, _stable_best_index(oracle_vals), None


def _binary_roc_curve_from_scores(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    y_true_arr = np.asarray([int(v) for v in y_true], dtype=int)
    y_score_arr = np.asarray([float(v) for v in y_score], dtype=float)

    mask = np.isfinite(y_score_arr)
    y_true_arr = y_true_arr[mask]
    y_score_arr = y_score_arr[mask]

    if y_true_arr.size == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), float("nan")

    pos = int(np.sum(y_true_arr == 1))
    neg = int(np.sum(y_true_arr == 0))
    if pos == 0 or neg == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), float("nan")

    order = np.argsort(-y_score_arr, kind="mergesort")
    y_true_sorted = y_true_arr[order]
    y_score_sorted = y_score_arr[order]

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    prev_score = None

    for i in range(y_true_sorted.size):
        score_i = float(y_score_sorted[i])
        if prev_score is not None and score_i != prev_score:
            tpr.append(tp / pos)
            fpr.append(fp / neg)

        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score_i

    tpr.append(tp / pos)
    fpr.append(fp / neg)

    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)
    auc = float(np.trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def _multiclass_state_metrics(
    per_state_rows: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    if len(per_state_rows) == 0:
        return {
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "macro_sensitivity": float("nan"),
            "macro_specificity": float("nan"),
            "balanced_accuracy": float("nan"),
        }

    action_ids = sorted(
        {str(r["oracle_best_action_id"]) for r in per_state_rows}
        | {str(r["predicted_best_action_id"]) for r in per_state_rows}
    )
    idx = {a: i for i, a in enumerate(action_ids)}
    cm = np.zeros((len(action_ids), len(action_ids)), dtype=float)

    for r in per_state_rows:
        i = idx[str(r["oracle_best_action_id"])]
        j = idx[str(r["predicted_best_action_id"])]
        cm[i, j] += 1.0

    total = float(cm.sum())
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    sensitivities: List[float] = []
    specificities: List[float] = []

    for k in range(len(action_ids)):
        tp = float(cm[k, k])
        fn = float(cm[k, :].sum() - tp)
        fp = float(cm[:, k].sum() - tp)
        tn = float(total - tp - fn - fp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        f1 = (2.0 * prec * rec / (prec + rec)) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        sensitivities.append(rec)
        specificities.append(spec)

    def _nanmean(xs: Sequence[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        if arr.size == 0 or np.all(~np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    macro_precision = _nanmean(precisions)
    macro_recall = _nanmean(recalls)
    macro_f1 = _nanmean(f1s)
    macro_sensitivity = _nanmean(sensitivities)
    macro_specificity = _nanmean(specificities)
    balanced_accuracy = (
        float(np.nanmean([macro_sensitivity, macro_specificity]))
        if np.isfinite(macro_sensitivity) or np.isfinite(macro_specificity)
        else float("nan")
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "balanced_accuracy": balanced_accuracy,
    }

def _multiclass_state_metrics_tie_aware(
    per_state_rows: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    if len(per_state_rows) == 0:
        return {
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "macro_sensitivity": float("nan"),
            "macro_specificity": float("nan"),
            "balanced_accuracy": float("nan"),
        }

    action_ids = sorted(
        {str(r["oracle_best_action_id"]) for r in per_state_rows}
        | {str(r["predicted_best_action_id"]) for r in per_state_rows}
    )
    idx = {a: i for i, a in enumerate(action_ids)}
    cm = np.zeros((len(action_ids), len(action_ids)), dtype=float)

    for r in per_state_rows:
        oracle_a = str(r["oracle_best_action_id"])
        pred_a = str(r["predicted_best_action_id"])
        tied_ids_raw = r.get("oracle_tied_best_action_ids_json", "")

        try:
            tied_ids = json.loads(tied_ids_raw) if str(tied_ids_raw).strip() != "" else [oracle_a]
        except Exception:
            tied_ids = [oracle_a]

        tied_ids = {str(a) for a in tied_ids}

        if oracle_a not in idx or pred_a not in idx:
            continue

        i = idx[oracle_a]

        if pred_a in tied_ids:
            j = i
            cm[i, j] += 1.0
        else:
            j = idx[pred_a]
            cm[i, j] += 1.0

    total = float(cm.sum())
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    sensitivities: List[float] = []
    specificities: List[float] = []

    for k in range(len(action_ids)):
        tp = float(cm[k, k])
        fn = float(cm[k, :].sum() - tp)
        fp = float(cm[:, k].sum() - tp)
        tn = float(total - tp - fn - fp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        f1 = (2.0 * prec * rec / (prec + rec)) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        sensitivities.append(rec)
        specificities.append(spec)

    def _nanmean(xs: Sequence[float]) -> float:
        arr = np.asarray(xs, dtype=float)
        if arr.size == 0 or np.all(~np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    macro_precision = _nanmean(precisions)
    macro_recall = _nanmean(recalls)
    macro_f1 = _nanmean(f1s)
    macro_sensitivity = _nanmean(sensitivities)
    macro_specificity = _nanmean(specificities)
    balanced_accuracy = (
        float(np.nanmean([macro_sensitivity, macro_specificity]))
        if np.isfinite(macro_sensitivity) or np.isfinite(macro_specificity)
        else float("nan")
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "balanced_accuracy": balanced_accuracy,
    }

def _classification_row_auc(
    per_row_records: Sequence[Dict[str, Any]],
    per_state_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    oracle_best_by_state: Dict[str, set[str]] = {}
    for r in per_state_rows:
        state_id = str(r["state_id"])
        tied_ids_raw = r.get("oracle_tied_best_action_ids_json", "")
        try:
            tied_ids = set(json.loads(tied_ids_raw)) if str(tied_ids_raw).strip() != "" else {str(r["oracle_best_action_id"])}
        except Exception:
            tied_ids = {str(r["oracle_best_action_id"])}
        oracle_best_by_state[state_id] = {str(x) for x in tied_ids}

    y_true: List[int] = []
    y_score: List[float] = []

    for row in per_row_records:
        state_id = str(row.get("state_id", ""))
        action_id = str(row.get("action_id", ""))
        if state_id not in oracle_best_by_state:
            continue
        score = float(row.get("pred_score", float("nan")))
        if not np.isfinite(score):
            continue
        y_true.append(1 if action_id in oracle_best_by_state[state_id] else 0)
        y_score.append(score)

    fpr, tpr, auc = _binary_roc_curve_from_scores(y_true, y_score)
    return {
        "y_true": y_true,
        "y_score": y_score,
        "fpr": fpr,
        "tpr": tpr,
        "auroc": auc,
        "n_positive_rows": int(sum(y_true)),
        "n_negative_rows": int(len(y_true) - sum(y_true)),
    }


def _plot_policy_auroc(
    roc_payload: Dict[str, Any],
    out_path: str,
) -> None:
    fpr = np.asarray(roc_payload.get("fpr", [0.0, 1.0]), dtype=float)
    tpr = np.asarray(roc_payload.get("tpr", [0.0, 1.0]), dtype=float)
    auroc = float(roc_payload.get("auroc", float("nan")))

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    if fpr.size < 2 or tpr.size < 2 or not np.isfinite(auroc):
        ax.axis("off")
        ax.text(0.5, 0.5, "AUROC unavailable", ha="center", va="center", fontsize=11)
    else:
        ax.plot(fpr, tpr, linewidth=1.6, color="#1f77b4", label=f"AUROC = {auroc:.3f}")
        ax.fill_between(fpr, 0.0, tpr, color="#1f77b4", alpha=0.25)
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="black")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("False positive rate", fontsize=11)
        ax.set_ylabel("True positive rate", fontsize=11)
        ax.set_title("Oracle-best row AUROC", fontsize=13, fontweight="bold")
        _nature_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)

def _write_txt_summary(path: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("POLICY SELECTOR EVALUATION")
    lines.append("=" * 80)
    for k in (
        "trained_dir",
        "data_folder",
        "split",
        "task",
        "oracle_metric",
        "selection_direction",
        "oracle_best_action_loss_mode",
        "n_states",
        "n_rows",
        "n_actions",
    ):
        if k in payload:
            lines.append(f"{k}: {payload[k]}")
    
    lines.append(f"tie_aware_policy_accuracy: {payload.get('policy_accuracy', float('nan'))}")
    lines.append(f"strict_policy_accuracy: {payload.get('strict_policy_accuracy', float('nan'))}")
    lines.append(f"tie_aware_top2_accuracy: {payload.get('top2_accuracy', float('nan'))}")
    lines.append(f"strict_top2_accuracy: {payload.get('strict_top2_accuracy', float('nan'))}")
    lines.append(f"strict_baseline_improvement_rate_gt0: {payload.get('baseline_improvement_rate', float('nan'))}")
    lines.append(f"tie_aware_baseline_non_worse_rate_ge0: {payload.get('baseline_non_worse_rate', float('nan'))}")
    lines.append(f"baseline_tie_rate_eq0: {payload.get('baseline_tie_rate', float('nan'))}")
    lines.append(f"tie_state_rate: {payload.get('tie_state_rate', float('nan'))}")
    lines.append("")
    lines.append("REGRET")
    lines.append("-" * 80)
    regret = payload.get("regret", {})
    for k in sorted(regret.keys()):
        lines.append(f"{k}: {regret[k]}")
    
    
    classification_metrics = payload.get("classification_metrics", {})
    if isinstance(classification_metrics, dict) and classification_metrics:
        lines.append("")
        lines.append("CLASSIFICATION METRICS")
        lines.append("-" * 80)
        tie_aware_classification_metrics = payload.get("tie_aware_classification_metrics", {})
        for k in (
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_sensitivity",
            "macro_specificity",
            "balanced_accuracy",
        ):
            if k in classification_metrics:
                lines.append(
                    f"strict_{k} (tie_aware={tie_aware_classification_metrics.get(k, float('nan'))}): {classification_metrics[k]}"
                )
        
        if "auroc" in classification_metrics:
            lines.append(f"auroc: {classification_metrics['auroc']}")
    lines.append("")
    lines.append("ACTION COUNTS")
    lines.append("-" * 80)

    display_pred = payload.get("predicted_action_display_counts", {})
    display_oracle = payload.get("oracle_action_display_counts", {})
    raw_pred = payload.get("predicted_action_counts", {})
    raw_oracle = payload.get("oracle_action_counts", {})
    action_catalog = payload.get("action_catalog", [])

    if isinstance(action_catalog, list) and action_catalog:
        for entry in action_catalog:
            if not isinstance(entry, dict):
                continue
            aid = str(entry.get("action_id", "")).strip()
            if aid == "":
                continue
            label = str(entry.get("action_display_name", entry.get("action_name", aid))).strip() or aid
            lines.append(f"predicted::{label}: {int(raw_pred.get(aid, 0))}")
        for entry in action_catalog:
            if not isinstance(entry, dict):
                continue
            aid = str(entry.get("action_id", "")).strip()
            if aid == "":
                continue
            label = str(entry.get("action_display_name", entry.get("action_name", aid))).strip() or aid
            lines.append(f"oracle::{label}: {int(raw_oracle.get(aid, 0))}")
    else:
        for name, cnt in sorted(display_pred.items()):
            lines.append(f"predicted::{name}: {cnt}")
        for name, cnt in sorted(display_oracle.items()):
            lines.append(f"oracle::{name}: {cnt}")

    if display_pred != raw_pred or display_oracle != raw_oracle:
        lines.append("")
        lines.append("ACTION COUNTS (RAW IDS)")
        lines.append("-" * 80)
        if isinstance(action_catalog, list) and action_catalog:
            for entry in action_catalog:
                if not isinstance(entry, dict):
                    continue
                aid = str(entry.get("action_id", "")).strip()
                if aid == "":
                    continue
                lines.append(f"predicted::{aid}: {int(raw_pred.get(aid, 0))}")
            for entry in action_catalog:
                if not isinstance(entry, dict):
                    continue
                aid = str(entry.get("action_id", "")).strip()
                if aid == "":
                    continue
                lines.append(f"oracle::{aid}: {int(raw_oracle.get(aid, 0))}")
        else:
            for name, cnt in sorted(raw_pred.items()):
                lines.append(f"predicted::{name}: {cnt}")
            for name, cnt in sorted(raw_oracle.items()):
                lines.append(f"oracle::{name}: {cnt}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate action ranking on a causal policy-manifest dataset.")
    parser.add_argument("--data_folder", required=True, type=str, help="Folder containing policy_manifest.csv and PT files.")
    parser.add_argument("--trained_dir", required=True, type=str, help="Training output dir containing trained_model.pt and metrics_summary.json.")
    parser.add_argument("--task", default="", type=str, help="Optional override. Otherwise read from trained_dir metrics_summary.json.")
    parser.add_argument("--split", default="test", type=str, choices=["train", "validation", "val", "test"], help="Manifest split to evaluate.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--oracle_metric", default="auto", type=str, help="Manifest column used for oracle action comparison, e.g. y_h7_endog_share. 'auto' maps from task.")
    parser.add_argument("--selection_direction", default="auto", type=str, choices=["auto", "minimize", "maximize"], help="Whether lower or higher values are better.")
    parser.add_argument("--use_action_conditioning", type=str2bool, default=None, help="Optional override of trained config.")
    parser.add_argument("--action_hidden_dim", type=int, default=None, help="Optional override of trained config.")
    parser.add_argument("--action_interaction_hidden_dim", type=int, default=None, help="Optional override of trained config.")
    parser.add_argument("--action_interaction_dropout", type=float, default=None, help="Optional override of trained config.")
    parser.add_argument("--require_complete_action_set", type=str2bool, default=True, help="If true, drop states with duplicate, missing, or extra actions relative to the expected action menu.")
    parser.add_argument("--emit_action_scores_csv", type=str2bool, default=True)
    parser.add_argument("--out_dir", default="policy_selector_eval", type=str)
    parser.add_argument(
    "--candidate_interventions_json",
        default="",
        type=str,
        help="Optional path to candidate_interventions.json. If provided, dashboard labels prefer display_name/label from that file.",
    )
    args = parser.parse_args()

    data_folder = os.path.abspath(args.data_folder)
    trained_dir = os.path.abspath(args.trained_dir)
    out_dir = os.path.abspath(args.out_dir)
    # Always start evaluation from a clean output directory so JSON/CSV/PNG
    # artifacts cannot be mixed with stale files from a previous run.
    shutil.rmtree(out_dir, ignore_errors=True)
    _ensure_dir(out_dir)

    split_name = "validation" if str(args.split).strip().lower() == "val" else str(args.split).strip().lower()
    candidate_label_map = _build_candidate_action_display_map(args.candidate_interventions_json)
    metrics_json = os.path.join(trained_dir, "metrics_summary.json")
    trained_summary = _load_json(metrics_json)
    trained_cfg = dict(trained_summary.get("config", {}))
    train_T = int(trained_cfg.get("T", 7))
    train_sliding_step = int(trained_cfg.get("sliding_step", 1))

    dataset = TemporalGraphDataset(
        folder=data_folder,
        T=train_T,
        sliding_step=train_sliding_step,
        prefer_pt_metadata=True,
        require_pt_metadata=True,
        fail_on_noncontiguous=True,
    )

    task_name = str(args.task).strip() or str(trained_summary.get("task", "")).strip()
    if task_name == "":
        raise ValueError("Could not resolve task name. Pass --task or ensure trained_dir/metrics_summary.json contains it.")
    task = _get_task(task_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trained_summary, task_name = _load_model_from_training_dir(
        dataset=dataset,
        trained_dir=trained_dir,
        task=task,
        device=device,
        task_name_override=task_name,
        use_action_conditioning_override=args.use_action_conditioning,
        action_hidden_dim_override=args.action_hidden_dim,
        action_interaction_hidden_dim_override=args.action_interaction_hidden_dim,
        action_interaction_dropout_override=args.action_interaction_dropout,
    )

    eval_idxs = _subset_indices_for_split(dataset, split_name)
    if len(eval_idxs) == 0:
        raise ValueError(f"No rows found for split='{split_name}' in policy manifest: {data_folder}")

    # Always also load train rows so Panel A can compare train vs test/train.
    extra_plot_splits = ["train"]
    plot_idxs: List[int] = []
    seen_plot_idx = set()

    for idx in eval_idxs:
        if idx not in seen_plot_idx:
            plot_idxs.append(idx)
            seen_plot_idx.add(idx)

    for extra_split in extra_plot_splits:
        extra_idxs = _subset_indices_for_split(dataset, extra_split)
        for idx in extra_idxs:
            if idx not in seen_plot_idx:
                plot_idxs.append(idx)
                seen_plot_idx.add(idx)

    subset = Subset(dataset, plot_idxs)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_temporal_graph_batch,
    )
    if str(args.oracle_metric).strip().lower() == "auto":
        oracle_metric, auto_oracle_direction = _auto_oracle_metric(task_name)
    else:
        oracle_metric = str(args.oracle_metric).strip()
        inferred_metric, inferred_direction = _auto_oracle_metric(task_name)
        if oracle_metric == inferred_metric:
            auto_oracle_direction = inferred_direction
        else:
            auto_oracle_direction = "minimize"

    # pred_score is converted to a task-aware utility in
    # _task_probability_and_prediction(), so higher is always better by default.
    pred_direction = (
        "maximize"
        if str(args.selection_direction).strip().lower() == "auto"
        else str(args.selection_direction).strip().lower()
    )
    oracle_direction = str(auto_oracle_direction).strip().lower()

    per_row_records: List[Dict[str, Any]] = []
    sample_ptr = 0

    model.eval()
    with torch.no_grad():
        for batched_graphs, labels_dict in loader:
            batched_graphs_dev = [g.to(device) for g in batched_graphs]
            labels_dev = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in labels_dict.items()
            }
            action_features = _extract_action_features(batched_graphs_dev, labels_dev, device=device)
            state_summary_features = _extract_state_summary_features(batched_graphs_dev, labels_dev, device=device)
            y_hat = model(
                batched_graphs_dev,
                action_features=action_features,
                state_summary_features=state_summary_features,
            )
            pred_score, pred_label_or_value = _task_probability_and_prediction(
                task=task,
                y_hat=y_hat,
                task_name=task_name,
                oracle_metric=oracle_metric,
                oracle_direction=oracle_direction,
            )

            batch_size = int(len(pred_score))
            for b in range(batch_size):
                dataset_index = int(plot_idxs[sample_ptr])
                sample_ptr += 1
                row_meta = _get_row_meta(dataset, dataset_index)

                rec: Dict[str, Any] = {
                    "dataset_index": dataset_index,
                    "split": str(row_meta.get("split", "")),
                    "state_id": str(row_meta.get("state_id", "")),
                    "pair_id": str(row_meta.get("pair_id", "")),
                    "seed": _safe_int(row_meta.get("seed", 0), 0),
                    "decision_day": _safe_int(row_meta.get("decision_day", 0), 0),
                    "action_id": str(row_meta.get("action_id", "")),
                    "action_name": str(row_meta.get("action_name", "")),
                    "action_index": _safe_int(row_meta.get("action_index", 10**9), 10**9),
                    "action_display_name": _make_action_display_name(
                        row_meta,
                        candidate_label_map=candidate_label_map,
                    ),
                    "is_baseline": _safe_int(row_meta.get("is_baseline", 0), 0),
                    "pred_score": float(pred_score[b]),
                    "pred_value": float(pred_label_or_value[b]) if not task.is_classification else int(pred_label_or_value[b]),
                    "oracle_metric": str(oracle_metric),
                    "oracle_value": _safe_float(row_meta.get(oracle_metric, float("nan")), float("nan")),
                }

                # Copy manifest oracle and target columns for downstream policy analysis.
                for k, v in row_meta.items():
                    key = str(k)
                    if (
                        key == "action_index"
                        or key.startswith("y_h")
                        or key.startswith("oracle_best_action_")
                        or key.startswith("oracle_tie_count_")
                        or key.startswith("is_oracle_best_")
                    ):
                        rec[key] = v

                per_row_records.append(rec)

    if sample_ptr != len(plot_idxs):
        raise RuntimeError(f"Prediction row alignment mismatch: scored={sample_ptr} expected={len(plot_idxs)}")
      
    # Group rows by state and compare action ranking.
    rows_by_state: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in per_row_records:
        rows_by_state[str(rec["state_id"])].append(rec)

    expected_action_ids_ordered, action_label_map_full, action_catalog = _build_ordered_action_catalog(
        dataset=dataset,
        per_row_records=per_row_records,
        candidate_label_map=candidate_label_map,
    )
    expected_action_ids = set(expected_action_ids_ordered)

    predicted_action_counts: Counter = Counter()
    oracle_action_counts: Counter = Counter()
    baseline_improvement_flags: List[int] = []
    baseline_tie_flags: List[int] = []
    regrets: List[float] = []
    per_state_rows: List[Dict[str, Any]] = []

    dropped_states: List[str] = []
    dropped_state_reasons: Dict[str, str] = {}
    for state_id, rows in sorted(rows_by_state.items()):
        rows_sorted, oracle_best_idx, oracle_error = _resolve_state_oracle_row(
            rows=rows,
            task_name=task_name,
            oracle_metric=oracle_metric,
            oracle_direction=oracle_direction,
        )
        action_ids = [str(r["action_id"]) for r in rows_sorted]
        unique_action_ids = set(action_ids)

        if len(unique_action_ids) != len(action_ids):
            if bool(args.require_complete_action_set):
                dropped_states.append(state_id)
                dropped_state_reasons[state_id] = "duplicate_actions"
                continue

        if bool(args.require_complete_action_set) and expected_action_ids and unique_action_ids != expected_action_ids:
            dropped_states.append(state_id)
            missing_action_ids = sorted(expected_action_ids - unique_action_ids)
            extra_action_ids = sorted(unique_action_ids - expected_action_ids)
            reason_bits: List[str] = []
            if missing_action_ids:
                reason_bits.append("missing=" + ",".join(missing_action_ids))
            if extra_action_ids:
                reason_bits.append("extra=" + ",".join(extra_action_ids))
            dropped_state_reasons[state_id] = ";".join(reason_bits) if reason_bits else "incomplete_action_set"
            continue

        if oracle_error is not None:
            dropped_states.append(state_id)
            dropped_state_reasons[state_id] = str(oracle_error)
            continue
        if oracle_best_idx is None:
            dropped_states.append(state_id)
            dropped_state_reasons[state_id] = "oracle_resolution_failed"
            continue

        pred_vals = np.asarray(_apply_direction([r["pred_score"] for r in rows_sorted], pred_direction), dtype=float)
        oracle_vals = np.asarray(_apply_direction([r["oracle_value"] for r in rows_sorted], oracle_direction), dtype=float)

        if np.any(~np.isfinite(oracle_vals)):
            dropped_states.append(state_id)
            dropped_state_reasons[state_id] = "nonfinite_oracle"
            continue

        pred_best_idx = _stable_best_index(pred_vals)

        oracle_tied_best_indices = set(_oracle_tied_best_indices(rows_sorted, oracle_direction))
        if len(oracle_tied_best_indices) == 0:
            oracle_tied_best_indices = {int(oracle_best_idx)}
        oracle_tied_best_action_ids = [str(rows_sorted[i]["action_id"]) for i in sorted(oracle_tied_best_indices)]

        # Top-2 membership for robustness when actions are close.
        top2_oracle = set(_stable_topk_indices(oracle_vals, 2))

        pred_best = rows_sorted[pred_best_idx]
        oracle_best = rows_sorted[oracle_best_idx]
        predicted_action_counts[str(pred_best["action_id"])] += 1
        oracle_action_counts[str(oracle_best["action_id"])] += 1

        baseline_row = None
        for r in rows_sorted:
            if int(r.get("is_baseline", 0)) == 1:
                baseline_row = r
                break

        regret = float(oracle_vals[pred_best_idx] - oracle_vals[oracle_best_idx])
        regret = max(0.0, regret)
        regrets.append(regret)

        baseline_improved = None
        baseline_equal = None
        baseline_delta = float("nan")
        if baseline_row is not None:
            baseline_idx = rows_sorted.index(baseline_row)
            if oracle_direction == "maximize":
                baseline_delta = float(rows_sorted[pred_best_idx]["oracle_value"] - rows_sorted[baseline_idx]["oracle_value"])
            else:
                baseline_delta = float(rows_sorted[baseline_idx]["oracle_value"] - rows_sorted[pred_best_idx]["oracle_value"])
            baseline_improved = 1 if baseline_delta > 0 else 0
            baseline_equal = 1 if math.isclose(float(baseline_delta), 0.0, rel_tol=1e-12, abs_tol=1e-12) else 0
            baseline_improvement_flags.append(int(baseline_improved))
            baseline_tie_flags.append(int(baseline_equal))

        oracle_tie_count = rows_sorted[0].get(f"oracle_tie_count_h{_extract_policy_horizon(task_name, oracle_metric) or 0}", "")
        if str(oracle_tie_count).strip() == "":
            best_val = float(rows_sorted[oracle_best_idx]["oracle_value"])
            oracle_tie_count = int(sum(
                1
                for row in rows_sorted
                if math.isclose(float(row.get("oracle_value", float("nan"))), best_val, rel_tol=1e-12, abs_tol=1e-12)
            ))
        else:
            oracle_tie_count = _safe_int(oracle_tie_count, 1)

        per_state_rows.append(
            {
                "state_id": state_id,
                "split": str(rows_sorted[0].get("split", "")),
                "seed": int(rows_sorted[0].get("seed", 0)),
                "decision_day": int(rows_sorted[0].get("decision_day", 0)),
                "n_actions": len(rows_sorted),
                "predicted_best_action_id": str(pred_best["action_id"]),
                "predicted_best_action_name": str(pred_best["action_name"]),
                "predicted_best_action_index": _safe_int(pred_best.get("action_index", 10**9), 10**9),
                "predicted_best_action_display_name": str(pred_best.get("action_display_name", pred_best["action_name"])),
                "predicted_best_pred_score": float(pred_best["pred_score"]),
                "predicted_best_oracle_value": float(pred_best["oracle_value"]),
                "oracle_best_action_id": str(oracle_best["action_id"]),
                "oracle_best_action_name": str(oracle_best["action_name"]),
                "oracle_best_action_index": _safe_int(oracle_best.get("action_index", 10**9), 10**9),
                "oracle_best_action_display_name": str(oracle_best.get("action_display_name", oracle_best["action_name"])),
                "oracle_best_pred_score": float(oracle_best["pred_score"]),
                "oracle_best_oracle_value": float(oracle_best["oracle_value"]),
                "oracle_tie_count": int(oracle_tie_count),
                "oracle_tied_best_action_ids_json": json.dumps(oracle_tied_best_action_ids, sort_keys=True),
                "policy_match": 1 if str(pred_best["action_id"]) in set(oracle_tied_best_action_ids) else 0,
                "strict_policy_match": 1 if str(pred_best["action_id"]) == str(oracle_best["action_id"]) else 0,
                "top2_match": 1 if (pred_best_idx in top2_oracle or pred_best_idx in oracle_tied_best_indices) else 0,
                "strict_top2_match": 1 if pred_best_idx in top2_oracle else 0,
                "regret": float(regret),
                "baseline_oracle_delta": float(baseline_delta),
                "baseline_improved": baseline_improved if baseline_improved is not None else "",
                "baseline_tie": baseline_equal if baseline_equal is not None else "",
                "baseline_action_id": "" if baseline_row is None else str(baseline_row["action_id"]),
                "baseline_action_display_name": "" if baseline_row is None else str(baseline_row.get("action_display_name", baseline_row["action_name"])),
                "baseline_oracle_value": float("nan") if baseline_row is None else float(baseline_row["oracle_value"]),
            }
        )

    if len(per_state_rows) == 0:
        raise RuntimeError("No evaluable states remained after filtering. Check manifest completeness and oracle metric availability.")

    all_evaluable_state_ids = {str(r["state_id"]) for r in per_state_rows}
    per_row_records_raw_all = [dict(r) for r in per_row_records]
    per_row_records = _filter_per_row_records_by_state_ids(per_row_records, all_evaluable_state_ids)

    metric_state_rows = _filter_per_state_rows_by_split(per_state_rows, split_name)
    metric_state_ids = {str(r["state_id"]) for r in metric_state_rows}
    metric_per_row_records = _filter_per_row_records_by_state_ids(per_row_records, metric_state_ids)
    raw_requested_split_records = _filter_per_row_records_by_split(per_row_records_raw_all, split_name)
    if len(metric_state_rows) == 0:
        raise RuntimeError(f"No evaluable states remained for requested split='{split_name}'.")

    policy_accuracy = float(np.mean([int(r["policy_match"]) for r in metric_state_rows]))
    strict_policy_accuracy = float(np.mean([int(r.get("strict_policy_match", r["policy_match"])) for r in metric_state_rows]))
    top2_accuracy = float(np.mean([int(r["top2_match"]) for r in metric_state_rows]))
    strict_top2_accuracy = float(np.mean([int(r.get("strict_top2_match", r["top2_match"])) for r in metric_state_rows]))
    tie_state_rate = float(np.mean([1 if int(r.get("oracle_tie_count", 1)) > 1 else 0 for r in metric_state_rows]))
    
    metric_baseline_improved = [
        int(r["baseline_improved"])
        for r in metric_state_rows
        if str(r.get("baseline_improved", "")) != ""
    ]
    metric_baseline_ties = [
        int(r["baseline_tie"])
        for r in metric_state_rows
        if str(r.get("baseline_tie", "")) != ""
    ]
    
    metric_baseline_non_worse = []
    for r in metric_state_rows:
        if str(r.get("baseline_improved", "")) == "" and str(r.get("baseline_tie", "")) == "":
            continue
        improved = int(r.get("baseline_improved", 0)) if str(r.get("baseline_improved", "")) != "" else 0
        tied = int(r.get("baseline_tie", 0)) if str(r.get("baseline_tie", "")) != "" else 0
        metric_baseline_non_worse.append(1 if (improved == 1 or tied == 1) else 0)
    
    baseline_improvement_rate = (
        float(np.mean(metric_baseline_improved))
        if metric_baseline_improved else float("nan")
    )
    baseline_tie_rate = (
        float(np.mean(metric_baseline_ties))
        if metric_baseline_ties else float("nan")
    )
    baseline_non_worse_rate = (
        float(np.mean(metric_baseline_non_worse))
        if metric_baseline_non_worse else float("nan")
    )

    classification_metrics: Dict[str, float] = {}
    tie_aware_classification_metrics: Dict[str, float] = {}
    
    if bool(task.is_classification):
        classification_metrics = _multiclass_state_metrics(metric_state_rows)
        tie_aware_classification_metrics = _multiclass_state_metrics_tie_aware(metric_state_rows)
    
        roc_payload = _classification_row_auc(
            per_row_records=metric_per_row_records,
            per_state_rows=metric_state_rows,
        )
        classification_metrics["auroc"] = float(roc_payload.get("auroc", float("nan")))
        tie_aware_classification_metrics["auroc"] = float(roc_payload.get("auroc", float("nan")))
        
    metric_predicted_action_counts: Counter = Counter()
    metric_oracle_action_counts: Counter = Counter()
    for row in metric_state_rows:
        metric_predicted_action_counts[str(row["predicted_best_action_id"])] += 1
        metric_oracle_action_counts[str(row["oracle_best_action_id"])] += 1

    predicted_action_counts_full = _zero_filled_action_counts(expected_action_ids_ordered, metric_predicted_action_counts)
    oracle_action_counts_full = _zero_filled_action_counts(expected_action_ids_ordered, metric_oracle_action_counts)
    predicted_action_display_counts = _zero_filled_display_counts(expected_action_ids_ordered, action_label_map_full, metric_predicted_action_counts)
    oracle_action_display_counts = _zero_filled_display_counts(expected_action_ids_ordered, action_label_map_full, metric_oracle_action_counts)

    summary = {
        "trained_dir": trained_dir,
        "data_folder": data_folder,
        "split": split_name,
        "task": task_name,
        "oracle_metric": oracle_metric,
        "selection_direction": pred_direction,
        "oracle_best_action_loss_mode": str(trained_cfg.get("oracle_best_action_loss_mode", "unknown")),
        "oracle_direction": oracle_direction,
        "n_rows": int(len(metric_per_row_records)),
        "n_rows_evaluable_all": int(len(per_row_records)),
        "n_rows_scored_all": int(len(per_row_records_raw_all)),
        "n_rows_scored_requested_split_raw": int(len(raw_requested_split_records)),
        "n_states": len(metric_state_rows),
        "n_actions": len(expected_action_ids),
        "dropped_states": len(dropped_states),
        "dropped_state_ids": sorted(dropped_states),
        "dropped_state_reasons": dict(sorted(dropped_state_reasons.items())),
        "policy_accuracy": policy_accuracy,
        "strict_policy_accuracy": strict_policy_accuracy,
        "top2_accuracy": top2_accuracy,
        "strict_top2_accuracy": strict_top2_accuracy,
        "tie_state_rate": tie_state_rate,
        "baseline_improvement_rate": baseline_improvement_rate,
        "baseline_tie_rate": baseline_tie_rate,
        "baseline_non_worse_rate": baseline_non_worse_rate,
        "regret": _summarize_regret([float(r["regret"]) for r in metric_state_rows]),
        "predicted_action_counts": dict(predicted_action_counts_full),
        "oracle_action_counts": dict(oracle_action_counts_full),
        "predicted_action_display_counts": dict(predicted_action_display_counts),
        "oracle_action_display_counts": dict(oracle_action_display_counts),
        "action_ids": list(expected_action_ids_ordered),
        "action_display_name_map": dict(action_label_map_full),
        "action_catalog": list(action_catalog),
        "is_classification": bool(task.is_classification),
        "classification_metrics": classification_metrics,
        "tie_aware_classification_metrics": tie_aware_classification_metrics,
    }

    _write_json(os.path.join(out_dir, "policy_selection_summary.json"), summary)
    _write_txt_summary(os.path.join(out_dir, "policy_selection_summary.txt"), summary)

    with open(os.path.join(out_dir, "policy_selection_per_state.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_state_rows[0].keys()))
        writer.writeheader()
        for row in per_state_rows:
            writer.writerow(row)

    with open(os.path.join(out_dir, "policy_selection_per_state_requested_split.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_state_rows[0].keys()))
        writer.writeheader()
        for row in metric_state_rows:
            writer.writerow(row)

    if bool(args.emit_action_scores_csv):
        fieldnames = sorted({k for row in per_row_records_raw_all for k in row.keys()})

        with open(os.path.join(out_dir, "policy_action_scores.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_row_records:
                writer.writerow(row)

        with open(os.path.join(out_dir, "policy_action_scores_requested_split.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in metric_per_row_records:
                writer.writerow(row)

        with open(os.path.join(out_dir, "policy_action_scores_all_rows.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_row_records_raw_all:
                writer.writerow(row)

        with open(os.path.join(out_dir, "policy_action_scores_requested_split_all_rows.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_requested_split_records:
                writer.writerow(row)

    # Publication-style summary plots
    _plot_action_selection_confusion(
        per_state_rows=metric_state_rows,
        out_path=os.path.join(out_dir, "policy_selection_confusion_matrix.png"),
        all_action_ids=expected_action_ids_ordered,
        action_label_map=action_label_map_full,
    )
    _plot_regret_histogram(
        per_state_rows=metric_state_rows,
        out_path=os.path.join(out_dir, "policy_regret_histogram.png"),
    )
    _plot_baseline_gain(
        per_state_rows=metric_state_rows,
        out_path=os.path.join(out_dir, "policy_gain_over_baseline.png"),
    )
    _plot_action_selection_counts(
        summary=summary,
        per_state_rows=per_state_rows,
        out_path=os.path.join(out_dir, "policy_action_selection_counts.png"),
        all_action_ids=expected_action_ids_ordered,
        action_label_map=action_label_map_full,
    )
    _plot_predicted_vs_oracle_scores(
        per_row_records=metric_per_row_records,
        out_path=os.path.join(out_dir, "policy_predicted_vs_oracle_scores.png"),
    )
    
    if bool(task.is_classification):
      _plot_policy_auroc(
          roc_payload=_classification_row_auc(
              per_row_records=metric_per_row_records,
              per_state_rows=metric_state_rows,
          ),
          out_path=os.path.join(out_dir, "policy_auroc_curve.png"),
      )
    
    _plot_policy_eval_dashboard(
        summary=summary,
        panel_a_state_rows=per_state_rows,
        metric_state_rows=metric_state_rows,
        metric_per_row_records=metric_per_row_records,
        per_state_rows=per_state_rows,
        out_path=os.path.join(out_dir, "policy_evaluation_dashboard.png"),
        all_action_ids=expected_action_ids_ordered,
        action_label_map=action_label_map_full,
    )

    print(
        f"POLICY_SELECTOR_EVAL_OK split={split_name} states={len(metric_state_rows)} "
        f"tie_policy_accuracy={policy_accuracy:.4f} strict_policy_accuracy={strict_policy_accuracy:.4f} "
        f"top2_accuracy={top2_accuracy:.4f} mean_regret={summary['regret']['mean']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
