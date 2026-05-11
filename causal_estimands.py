#!/usr/bin/env python3
from __future__ import annotations

"""
causal_estimands.py

Stage 1 simulator-world causal estimands and polished publication-quality plots.

This script compares paired factual/counterfactual GraphML trajectories using the
causal metadata exported by generate_amr_data.py / convert_to_pt.py.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx


PREFERRED_GRAPH_KEYS: Tuple[str, ...] = (
    "future_endogenous_total",
    "future_importation_total",
    "future_selection_total",
    "future_transmission_total",
    "future_resistant_fraction",
    "future_resistant_total",
    "label_endogenous_importation_majority",
    "new_import_cr_total",
    "new_import_cs_total",
    "new_import_ir_total",
    "new_import_is_total",
    "new_trans_cr_total",
    "new_select_cr_total",
    "new_cr_acq_total",
    "new_ir_inf_total",
    "resistant_fraction",
)

FALLBACK_NODE_STATE_KEYS: Tuple[str, ...] = (
    "amr_state",
    "state",
    "obs_status",
    "observed_positive",
)



def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f



def _bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    import random

    rng = random.Random(1337)
    boots: List[float] = []
    n = len(values)
    for _ in range(max(100, int(n_boot))):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(boots)) - 1)
    hi_idx = min(len(boots) - 1, int((1.0 - alpha / 2.0) * len(boots)) - 1)
    return boots[lo_idx], boots[hi_idx]



def _read_graphml(path: Path) -> nx.Graph:
    return nx.read_graphml(path)



def _extract_node_based_resistant_total(g: nx.Graph) -> Optional[float]:
    total = 0.0
    found = False
    for _, attrs in g.nodes(data=True):
        for key in FALLBACK_NODE_STATE_KEYS:
            val = attrs.get(key)
            fval = _safe_float(val)
            sval = str(val).strip().upper()
            if sval in {"CR", "IR"} or (fval is not None and int(fval) in {2, 4}):
                total += 1.0
                found = True
                break
    return total if found else None



def _extract_graph_metrics(g: nx.Graph) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    graph_attrs = dict(g.graph)

    for key in PREFERRED_GRAPH_KEYS:
        val = _safe_float(graph_attrs.get(key))
        if val is not None:
            metrics[key] = val

    # Align simulator daily graph keys with causal-estimand names expected downstream.
    import_total = 0.0
    has_import = False
    for key in ("future_importation_total", "new_import_cr_total", "new_import_cs_total", "new_import_ir_total", "new_import_is_total"):
        val = _safe_float(graph_attrs.get(key))
        if val is not None:
            if key == "future_importation_total":
                metrics["future_importation_total"] = val
                has_import = True
            else:
                import_total += val
                has_import = True
    if has_import and "future_importation_total" not in metrics:
        metrics["future_importation_total"] = import_total

    trans_val = _safe_float(graph_attrs.get("future_transmission_total"))
    if trans_val is None:
        trans_val = _safe_float(graph_attrs.get("new_trans_cr_total"))
    if trans_val is not None:
        metrics["future_transmission_total"] = trans_val

    sel_val = _safe_float(graph_attrs.get("future_selection_total"))
    if sel_val is None:
        sel_val = _safe_float(graph_attrs.get("new_select_cr_total"))
    if sel_val is not None:
        metrics["future_selection_total"] = sel_val

    if "future_endogenous_total" not in metrics:
        trans = metrics.get("future_transmission_total")
        sel = metrics.get("future_selection_total")
        if trans is not None or sel is not None:
            metrics["future_endogenous_total"] = float((trans or 0.0) + (sel or 0.0))

    if "future_resistant_fraction" not in metrics:
        frac = _safe_float(graph_attrs.get("resistant_fraction"))
        if frac is not None:
            metrics["future_resistant_fraction"] = frac

    if "future_resistant_total" not in metrics:
        node_based = _extract_node_based_resistant_total(g)
        if node_based is not None:
            metrics["future_resistant_total"] = node_based

    if "future_resistant_total" not in metrics:
        total = _safe_float(graph_attrs.get("new_cr_acq_total"))
        if total is not None:
            metrics["future_resistant_total"] = total

    # Majority label from aligned endogenous vs importation totals when not present.
    if "label_endogenous_importation_majority" not in metrics:
        endog = metrics.get("future_endogenous_total")
        imp = metrics.get("future_importation_total")
        if endog is not None and imp is not None and (endog + imp) > 0.0:
            metrics["label_endogenous_importation_majority"] = 1.0 if endog >= imp else 0.0

    metrics["num_nodes"] = float(g.number_of_nodes())
    metrics["num_edges"] = float(g.number_of_edges())
    return metrics



def _index_graphml(folder: Path) -> Dict[Tuple[str, str, str], Path]:
    indexed: Dict[Tuple[str, str, str], Path] = {}
    for path in sorted(folder.rglob("*.graphml")):
        g = _read_graphml(path)
        pair_id = str(g.graph.get("cf_pair_id", "")).strip()
        day = str(g.graph.get("day", "")).strip()
        region = str(g.graph.get("region", "")).strip()
        if not pair_id:
            pair_id = path.parent.name
        if not day:
            day = path.stem
        indexed[(pair_id, day, region)] = path
    return indexed



def paired_metric_rows(factual_dir: Path, counterfactual_dir: Path) -> List[Dict[str, Any]]:
    factual_idx = _index_graphml(factual_dir)
    counterfactual_idx = _index_graphml(counterfactual_dir)
    shared_keys = sorted(set(factual_idx).intersection(counterfactual_idx))
    rows: List[Dict[str, Any]] = []

    for pair_key in shared_keys:
        f_graph = _read_graphml(factual_idx[pair_key])
        c_graph = _read_graphml(counterfactual_idx[pair_key])
        f_metrics = _extract_graph_metrics(f_graph)
        c_metrics = _extract_graph_metrics(c_graph)
        metric_names = sorted(set(f_metrics).intersection(c_metrics))

        row: Dict[str, Any] = {
            "cf_pair_id": pair_key[0],
            "day": pair_key[1],
            "region": pair_key[2],
            "intervention_name": str(c_graph.graph.get("cf_intervention_name", "")).strip(),
            "intervention_target_type": str(c_graph.graph.get("cf_intervention_target_type", "")).strip(),
            "intervention_target_id": str(c_graph.graph.get("cf_intervention_target_id", "")).strip(),
        }
        for metric_name in metric_names:
            f_val = float(f_metrics[metric_name])
            c_val = float(c_metrics[metric_name])
            row[f"factual__{metric_name}"] = f_val
            row[f"counterfactual__{metric_name}"] = c_val
            row[f"delta__{metric_name}"] = c_val - f_val
        rows.append(row)
    return rows



def summarise_paired_effects(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n_pairs": 0, "metrics": {}}

    delta_keys = sorted({k for row in rows for k in row.keys() if str(k).startswith("delta__")})
    out: Dict[str, Any] = {
        "n_pairs": len(rows),
        "metrics": {},
    }

    for delta_key in delta_keys:
        metric_name = delta_key.replace("delta__", "", 1)
        values = [float(row[delta_key]) for row in rows if _safe_float(row.get(delta_key)) is not None]
        if not values:
            continue
        ate = sum(values) / len(values)
        ci_lo, ci_hi = _bootstrap_ci(values)
        harm_frac = sum(1 for v in values if v > 0) / len(values)
        benefit_frac = sum(1 for v in values if v < 0) / len(values)
        zero_frac = sum(1 for v in values if v == 0) / len(values)
        sd = 0.0 if len(values) == 1 else (sum((v - ate) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        out["metrics"][metric_name] = {
            "n": len(values),
            "ate": ate,
            "median_effect": median(values),
            "sd": sd,
            "min": min(values),
            "max": max(values),
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "fraction_harmful": harm_frac,
            "fraction_beneficial": benefit_frac,
            "fraction_zero": zero_frac,
        }
    return out



def write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))



def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")



def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )



def plot_effect_forest(summary: Mapping[str, Any], out_path: Path, title: str) -> None:
    metrics = summary.get("metrics", {})
    if not metrics:
        return
    _apply_publication_style()
    names = list(metrics.keys())
    ates = [metrics[k]["ate"] for k in names]
    lows = [metrics[k]["ci95_low"] if metrics[k]["ci95_low"] is not None else metrics[k]["ate"] for k in names]
    highs = [metrics[k]["ci95_high"] if metrics[k]["ci95_high"] is not None else metrics[k]["ate"] for k in names]
    y = list(range(len(names)))

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.7 * len(names))))
    ax.axvline(0.0, linewidth=1.2, linestyle="--")
    ax.errorbar(
        ates,
        y,
        xerr=[[a - l for a, l in zip(ates, lows)], [h - a for a, h in zip(ates, highs)]],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Average treatment effect (counterfactual - factual)")
    ax.set_title(title)
    ax.grid(axis="x", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)



def plot_effect_histogram(rows: Sequence[Mapping[str, Any]], metric_name: str, out_path: Path, title: str) -> None:
    delta_key = f"delta__{metric_name}"
    values = [float(row[delta_key]) for row in rows if _safe_float(row.get(delta_key)) is not None]
    if not values:
        return
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(30, max(10, int(len(values) ** 0.5) + 3)), edgecolor="black", linewidth=0.6)
    ax.axvline(0.0, linewidth=1.2, linestyle="--")
    ax.set_xlabel(f"Delta {metric_name} (counterfactual - factual)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)



def plot_sign_balance(summary: Mapping[str, Any], out_path: Path, title: str) -> None:
    metrics = summary.get("metrics", {})
    if not metrics:
        return
    _apply_publication_style()
    names = list(metrics.keys())
    harmful = [metrics[k]["fraction_harmful"] for k in names]
    beneficial = [metrics[k]["fraction_beneficial"] for k in names]
    neutral = [metrics[k]["fraction_zero"] for k in names]
    y = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.7 * len(names))))
    ax.barh(y, beneficial, label="Beneficial")
    ax.barh(y, neutral, left=beneficial, label="Neutral")
    ax.barh(y, harmful, left=[b + n for b, n in zip(beneficial, neutral)], label="Harmful")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of paired windows")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)



def generate_publication_figures(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out_dir: Path, stem: str = "causal") -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    forest_path = out_dir / f"{stem}_forest_ate.png"
    plot_effect_forest(summary, forest_path, title="Simulator-world causal effects")
    if forest_path.exists():
        generated.append(forest_path)

    sign_path = out_dir / f"{stem}_sign_balance.png"
    plot_sign_balance(summary, sign_path, title="Effect-direction balance across paired windows")
    if sign_path.exists():
        generated.append(sign_path)

    preferred = [
        "future_endogenous_total",
        "future_importation_total",
        "future_selection_total",
        "future_transmission_total",
        "future_resistant_total",
    ]
    available = [name for name in preferred if f"delta__{name}" in {k for row in rows for k in row.keys()}]
    for metric_name in available[:3]:
        out_path = out_dir / f"{stem}_hist_{metric_name}.png"
        plot_effect_histogram(rows, metric_name, out_path, title=f"Paired effects for {metric_name}")
        if out_path.exists():
            generated.append(out_path)
    return generated



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factual_dir", required=True, type=str)
    parser.add_argument("--counterfactual_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--stem", type=str, default="causal")
    args = parser.parse_args()

    factual_dir = Path(args.factual_dir).resolve()
    counterfactual_dir = Path(args.counterfactual_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = paired_metric_rows(factual_dir, counterfactual_dir)
    summary = summarise_paired_effects(rows)

    write_rows_csv(rows, out_dir / f"{args.stem}_paired_rows.csv")
    write_summary_json(summary, out_dir / f"{args.stem}_summary.json")
    generated = generate_publication_figures(rows, summary, out_dir, stem=args.stem)

    manifest = {
        "factual_dir": str(factual_dir),
        "counterfactual_dir": str(counterfactual_dir),
        "n_rows": len(rows),
        "generated_figures": [str(p) for p in generated],
    }
    write_summary_json(manifest, out_dir / f"{args.stem}_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
