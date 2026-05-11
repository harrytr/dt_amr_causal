#!/usr/bin/env python3
"""
Standalone feature-ablation runner for the AMR causal policy pipeline.


python run_policy_feature_ablations.py \
  --project_root "/Users/harrytriantafyllidis/AMR_MILP" \
  --data_folder "/Users/harrytriantafyllidis/AMR_MILP/experiments_causal_results/TRACK_ground_truth/work/causal_policy_monthly_shock_superspreader" \
  --out_root "/Users/harrytriantafyllidis/AMR_MILP/ablation_runs" \
  --task oracle_best_action_h7 \
  --oracle_metric auto \
  --candidate_interventions_json "/Users/harrytriantafyllidis/AMR_MILP/candidate_interventions.json" \
  --primary_metric policy_accuracy \
  --tolerance 0.01 \
  --train_extra "--T 7 --sliding_step 1 --hidden 32 --heads 2 --dropout 0.2 --transformer_layers 2 --sage_layers 2 --batch_size 16 --epochs 50 --lr 1e-4 --neighbor_sampling false --num_neighbors 15,10 --seed_count 256 --seed_strategy random --seed_batch_size 64 --max_sub_batches 4 --use_action_conditioning true --aux_policy_loss false --pairwise_policy_ranking_loss false --oracle_best_action_loss true --emit_translational_figures false"

What it does
------------
- Leaves the existing codebase untouched.
- Uses the existing dataset and runs from C2 onward only:
  1) train_amr_dygformer.py
  2) evaluate_policy_selector.py
- Applies feature ablations at runtime via monkey-patching inside an isolated
  worker process.
- Aggregates results across ablations.
- Identifies the smallest tested feature set that stays within a user-defined
  tolerance of the best primary metric.
- Writes CSV/JSON summaries and a publication-ready LaTeX table.

Typical use
-----------
python run_policy_feature_ablations.py \
  --project_root /path/to/AMR_V2-MILP_CLS_PRO \
  --data_folder /path/to/causal_policy_dataset \
  --out_root /path/to/ablation_results \
  --task oracle_best_action_h14 \
  --oracle_metric auto \
  --candidate_interventions_json /path/to/candidate_interventions.json \
  --train_extra "--T 14 --sliding_step 1 --hidden 32 --heads 2 --dropout 0.2 --transformer_layers 2 --sage_layers 2 --batch_size 16 --epochs 50 --lr 1e-4 --neighbor_sampling false --seed_count 256 --seed_strategy random --seed_batch_size 64 --max_sub_batches 4 --use_action_conditioning true --aux_policy_loss false --pairwise_policy_ranking_loss false --oracle_best_action_loss true --emit_translational_figures false"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NODE_FEATURES_GROUND_TRUTH: List[str] = [
    "is_staff",
    "amr_state_0",
    "amr_state_1",
    "amr_state_2",
    "amr_state_3",
    "amr_state_4",
    "abx_class",
    "is_isolated",
    "new_cr_acq_today",
    "new_ir_inf_today",
    "ward_id_norm",
    "ward_cover_norm",
    "obs_positive",
    "obs_known",
    "screened_today",
    "days_since_last_test_norm",
    "pending_test_days_norm",
    "pending_test_result",
    "needs_admission_screen",
    "present_today",
    "isolation_days_remaining_norm",
    "admission_age_norm",
]

NODE_FEATURES_PARTIAL_OBS: List[str] = [
    "is_staff",
    "obs_positive",
    "abx_class",
    "is_isolated",
    "new_cr_acq_today",
    "new_ir_inf_today",
    "ward_id_norm",
    "ward_cover_norm",
    "obs_known",
    "screened_today",
    "days_since_last_test_norm",
    "pending_test_days_norm",
    "needs_admission_screen",
    "present_today",
    "isolation_days_remaining_norm",
    "admission_age_norm",
]

EDGE_FEATURES: List[str] = [
    "edge_weight",
    "edge_type",
]

ACTION_FEATURES: List[str] = [
    "family_baseline",
    "family_set_screening_frequency",
    "family_set_screening_delay",
    "family_disable_isolation_response",
    "family_set_isolation_parameters",
    "family_reduce_ward_importation",
    "family_remove_staff_crossward_cover",
    "family_remove_specific_staff",
    "family_remove_edge",
    "family_other",
    "target_none",
    "target_global",
    "target_ward",
    "target_staff",
    "target_edge",
    "target_patient",
    "target_region",
    "target_hospital",
    "is_baseline",
    "policy_valid",
    "screen_freq_norm",
    "screen_delay_norm",
    "multiplier_encoded",
    "multiplier_cr_encoded",
    "multiplier_cs_encoded",
    "isolation_mult_encoded",
    "isolation_days_norm",
    "screen_on_admission",
    "start_day_norm",
    "end_day_norm",
    "action_name_hash",
    "action_id_hash",
    "target_hash",
]

STATE_FEATURES: List[str] = [
    "node_feat_mean_1",
    "node_feat_mean_2",
    "node_feat_mean_3",
    "node_feat_mean_4",
    "node_feat_mean_5",
    "node_feat_mean_6",
    "node_feat_std_1",
    "node_feat_std_2",
    "node_feat_std_3",
    "node_feat_std_4",
    "node_feat_std_5",
    "node_feat_std_6",
    "staff_frac",
    "patient_frac",
    "state_mean",
    "state_std",
    "state_min",
    "state_max",
    "state_positive_frac",
    "abx_positive_frac",
    "iso_positive_frac",
    "new_cr_sum_per_node",
    "new_ir_sum_per_node",
    "num_nodes_log_scaled",
    "edge_count_log_scaled",
    "mean_out_degree_log_scaled",
    "density",
    "edge_weight_mean_squashed",
    "edge_weight_std_squashed",
    "edge_type_entropy",
    "obs_positive_mean",
    "obs_known_mean",
    "screened_today_mean",
    "days_since_last_test_mean",
    "pending_test_days_mean",
    "pending_test_result_mean",
    "needs_admission_screen_mean",
    "present_today_mean",
    "isolation_days_remaining_mean",
    "admission_age_mean",
    "op_screen_every_k_days",
    "op_weekly_screen_day",
    "op_screen_on_admission",
    "op_screen_result_delay_days",
    "op_isolation_mult",
    "op_isolation_days",
    "op_persist_observations",
    "op_is_screening_day",
    "op_days_until_next_screen",
]


ACTION_BLOCKS: Dict[str, List[str]] = {
    "family": ACTION_FEATURES[0:10],
    "target": ACTION_FEATURES[10:18],
    "flags": ACTION_FEATURES[18:20],
    "numeric": ACTION_FEATURES[20:30],
    "hash": ACTION_FEATURES[30:33],
}

STATE_BLOCKS: Dict[str, List[str]] = {
    "node_moments": STATE_FEATURES[0:12],
    "burden_topology": STATE_FEATURES[12:30],
    "observation_ops": STATE_FEATURES[30:40],
    "operational_context": STATE_FEATURES[40:49],
}

NODE_BLOCKS_GROUND_TRUTH: Dict[str, List[str]] = {
    "role_state": NODE_FEATURES_GROUND_TRUTH[0:6],
    "burden": NODE_FEATURES_GROUND_TRUTH[6:10],
    "ward": NODE_FEATURES_GROUND_TRUTH[10:12],
    "observation_ops": NODE_FEATURES_GROUND_TRUTH[12:22],
}

NODE_BLOCKS_PARTIAL_OBS: Dict[str, List[str]] = {
    "role_state": NODE_FEATURES_PARTIAL_OBS[0:2],
    "burden": NODE_FEATURES_PARTIAL_OBS[2:6],
    "ward": NODE_FEATURES_PARTIAL_OBS[6:8],
    "observation_ops": NODE_FEATURES_PARTIAL_OBS[8:16],
}

FEATURE_EXPLANATIONS: Dict[str, str] = {
    "normal": "Temporal graph features passed through the node and edge encoders.",
    "state": "Explicit handcrafted state-summary vector derived from the current decision-state graph.",
    "intervention": "Explicit action/intervention vector derived from the policy-manifest row.",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    keep_node_gt: Tuple[str, ...]
    keep_node_po: Tuple[str, ...]
    keep_edge: Tuple[str, ...]
    keep_state: Tuple[str, ...]
    keep_action: Tuple[str, ...]
    use_action_conditioning: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "keep_node_gt": list(self.keep_node_gt),
            "keep_node_po": list(self.keep_node_po),
            "keep_edge": list(self.keep_edge),
            "keep_state": list(self.keep_state),
            "keep_action": list(self.keep_action),
            "use_action_conditioning": bool(self.use_action_conditioning),
        }


def _latex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = s
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        z = float(x)
        if math.isfinite(z):
            return z
    except Exception:
        pass
    return default


def _bool_to_cli(value: bool) -> str:
    return "true" if bool(value) else "false"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _default_variants() -> List[VariantSpec]:
    all_node_gt = tuple(NODE_FEATURES_GROUND_TRUTH)
    all_node_po = tuple(NODE_FEATURES_PARTIAL_OBS)
    all_edge = tuple(EDGE_FEATURES)
    all_state = tuple(STATE_FEATURES)
    all_action = tuple(ACTION_FEATURES)

    action_no_hash = tuple([f for f in ACTION_FEATURES if f not in ACTION_BLOCKS["hash"]])
    state_no_op = tuple([f for f in STATE_FEATURES if f not in STATE_BLOCKS["operational_context"]])
    action_family_only = tuple(ACTION_BLOCKS["family"])
    action_family_numeric = tuple(ACTION_BLOCKS["family"] + ACTION_BLOCKS["numeric"])

    node_core_gt = tuple(
        NODE_BLOCKS_GROUND_TRUTH["role_state"] + NODE_BLOCKS_GROUND_TRUTH["burden"] + NODE_BLOCKS_GROUND_TRUTH["ward"]
    )
    node_core_po = tuple(
        NODE_BLOCKS_PARTIAL_OBS["role_state"] + NODE_BLOCKS_PARTIAL_OBS["burden"] + NODE_BLOCKS_PARTIAL_OBS["ward"]
    )

    variants = [
        VariantSpec(
            name="full",
            description="All currently engineered features.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=all_state,
            keep_action=all_action,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="no_state_summary",
            description="Remove the explicit handcrafted state-summary branch entirely.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=tuple(),
            keep_action=all_action,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="no_action_hashes",
            description="Drop hashed action identity features only.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=all_state,
            keep_action=action_no_hash,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="no_state_op_context",
            description="Drop only operational-context features from the state-summary branch.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=state_no_op,
            keep_action=all_action,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="action_family_only",
            description="Use only intervention-family indicators in the action branch.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=all_state,
            keep_action=action_family_only,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="action_family_numeric",
            description="Keep family plus numeric intervention parameters; remove target IDs, flags, and hashes.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=all_state,
            keep_action=action_family_numeric,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="lean_no_state_no_hash",
            description="No state-summary branch and no action hashes.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=tuple(),
            keep_action=action_no_hash,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="lean_action_family_numeric_no_state",
            description="No state-summary branch, action family plus numeric parameters only.",
            keep_node_gt=all_node_gt,
            keep_node_po=all_node_po,
            keep_edge=all_edge,
            keep_state=tuple(),
            keep_action=action_family_numeric,
            use_action_conditioning=True,
        ),
        VariantSpec(
            name="node_core_plus_action_family_numeric_no_state",
            description="Core node features only, no state-summary branch, and compact action encoding.",
            keep_node_gt=node_core_gt,
            keep_node_po=node_core_po,
            keep_edge=all_edge,
            keep_state=tuple(),
            keep_action=action_family_numeric,
            use_action_conditioning=True,
        ),
    ]
    return variants


def _resolve_variants(names: Optional[Sequence[str]]) -> List[VariantSpec]:
    catalog = {v.name: v for v in _default_variants()}
    if not names:
        return list(catalog.values())
    selected: List[VariantSpec] = []
    for name in names:
        key = str(name).strip()
        if key not in catalog:
            raise KeyError(
                f"Unknown variant '{key}'. Available variants: {', '.join(sorted(catalog.keys()))}"
            )
        selected.append(catalog[key])
    return selected


def _count_dims(spec: VariantSpec) -> Dict[str, int]:
    return {
        "node_gt_dim": len(spec.keep_node_gt),
        "node_po_dim": len(spec.keep_node_po),
        "edge_dim": len(spec.keep_edge),
        "state_dim": len(spec.keep_state),
        "action_dim": len(spec.keep_action) if spec.use_action_conditioning else 0,
        "total_dim_gt": len(spec.keep_node_gt) + len(spec.keep_edge) + len(spec.keep_state) + (len(spec.keep_action) if spec.use_action_conditioning else 0),
        "total_dim_po": len(spec.keep_node_po) + len(spec.keep_edge) + len(spec.keep_state) + (len(spec.keep_action) if spec.use_action_conditioning else 0),
    }




def _extract_cli_value(cli_text: str, flag_name: str) -> Optional[str]:
    tokens = shlex.split(str(cli_text).strip())
    for i, tok in enumerate(tokens):
        if tok == flag_name and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(flag_name + '='):
            return tok.split('=', 1)[1]
    return None


def _resolve_train_window_params(train_extra: str) -> Tuple[int, int]:
    raw_T = _extract_cli_value(train_extra, '--T')
    raw_step = _extract_cli_value(train_extra, '--sliding_step')
    T = int(raw_T) if raw_T is not None else 7
    sliding_step = int(raw_step) if raw_step is not None else 1
    return T, sliding_step

def _discover_state_mode(project_root: Path, data_folder: Path, T: int, sliding_step: int) -> str:
    sys.path.insert(0, str(project_root))
    try:
        from temporal_graph_dataset import TemporalGraphDataset  # type: ignore
    finally:
        if sys.path and sys.path[0] == str(project_root):
            sys.path.pop(0)

    dataset = TemporalGraphDataset(
        folder=str(data_folder),
        T=int(T),
        sliding_step=int(sliding_step),
        prefer_pt_metadata=True,
        require_pt_metadata=True,
        fail_on_noncontiguous=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"Dataset appears empty: {data_folder}")
    graphs, _ = dataset[0]
    if not graphs:
        raise ValueError(f"Could not read any graphs from dataset: {data_folder}")
    x = getattr(graphs[-1], "x", None)
    if x is None or len(getattr(x, "shape", [])) != 2:
        raise ValueError("Could not infer node feature width from dataset sample.")
    feat_dim = int(x.shape[1])
    if feat_dim >= 22:
        return "ground_truth"
    if feat_dim >= 16:
        return "partial_observation"
    return "unknown"


def _build_train_cli_args(args: argparse.Namespace, variant: VariantSpec, train_out_dir: Path) -> List[str]:
    cli = [
        "--data_folder", str(args.data_folder),
        "--task", str(args.task),
        "--out_dir", str(train_out_dir),
        "--train_model", "true",
        "--use_action_conditioning", _bool_to_cli(variant.use_action_conditioning),
    ]
    cli.extend(shlex.split(str(args.train_extra).strip()))
    return cli


def _build_eval_cli_args(args: argparse.Namespace, variant: VariantSpec, train_out_dir: Path, eval_out_dir: Path) -> List[str]:
    cli = [
        "--data_folder", str(args.data_folder),
        "--trained_dir", str(train_out_dir),
        "--task", str(args.task),
        "--split", str(args.eval_split),
        "--oracle_metric", str(args.oracle_metric),
        "--selection_direction", str(args.selection_direction),
        "--out_dir", str(eval_out_dir),
        "--emit_action_scores_csv", "true",
        "--require_complete_action_set", "true",
        "--use_action_conditioning", _bool_to_cli(variant.use_action_conditioning),
    ]
    if str(args.candidate_interventions_json).strip() != "":
        cli.extend(["--candidate_interventions_json", str(args.candidate_interventions_json)])
    cli.extend(shlex.split(str(args.eval_extra).strip()))
    return cli


def _render_feature_keep_description(spec: VariantSpec, state_mode: str) -> str:
    node_dim = len(spec.keep_node_gt) if state_mode == "ground_truth" else len(spec.keep_node_po)
    pieces = [
        f"normal={node_dim} node + {len(spec.keep_edge)} edge",
        f"state={len(spec.keep_state)}",
        f"intervention={(len(spec.keep_action) if spec.use_action_conditioning else 0)}",
    ]
    return ", ".join(pieces)


def _generate_latex_table(rows: List[Dict[str, Any]], state_mode: str, primary_metric: str, tolerance: float) -> str:
    header = textwrap.dedent(
        rf"""
        % Requires: \usepackage{{booktabs}}, \usepackage{{longtable}}
        \begin{{longtable}}{{p{{4.8cm}}p{{5.7cm}}rrrrrr}}
        \caption{{Feature ablation study for the causal policy selector. The selected minimal architecture is the smallest tested variant whose { _latex_escape(primary_metric) } remains within {tolerance:.4f} of the best observed value.}}\\
        \label{{tab:policy_feature_ablation}}\\
        \toprule
        \textbf{{Variant}} & \textbf{{Description}} & \textbf{{Normal}} & \textbf{{State}} & \textbf{{Interv.}} & \textbf{{Primary}} & \textbf{{Top-2}} & \textbf{{Regret}} \\
        \midrule
        \endfirsthead
        \multicolumn{{8}}{{l}}{{\textit{{Table \thetable\ continued from previous page}}}}\\
        \toprule
        \textbf{{Variant}} & \textbf{{Description}} & \textbf{{Normal}} & \textbf{{State}} & \textbf{{Interv.}} & \textbf{{Primary}} & \textbf{{Top-2}} & \textbf{{Regret}} \\
        \midrule
        \endhead
        \bottomrule
        \endfoot
        """
    ).strip("\n")

    body_lines: List[str] = []
    for row in rows:
        normal_dim = row.get("node_gt_dim", 0) + row.get("edge_dim", 0) if state_mode == "ground_truth" else row.get("node_po_dim", 0) + row.get("edge_dim", 0)
        state_dim = row.get("state_dim", 0)
        action_dim = row.get("action_dim", 0)
        primary = _safe_float(row.get("primary_metric_value"))
        top2 = _safe_float(row.get("top2_accuracy"))
        regret = _safe_float(row.get("mean_regret"))
        variant_label = _latex_escape(row.get("variant", ""))
        if bool(row.get("is_selected_minimal", False)):
            variant_label = r"\textbf{" + variant_label + "}"
        description = _latex_escape(row.get("description", ""))
        body_lines.append(
            f"{variant_label} & {description} & {int(normal_dim)} & {int(state_dim)} & {int(action_dim)} & {primary:.4f} & {top2:.4f} & {regret:.4f} \\"
        )

    footer = "\\end{longtable}\n"
    return header + "\n" + "\n".join(body_lines) + "\n" + footer


def _write_feature_catalog(path: Path, variants: Sequence[VariantSpec]) -> None:
    payload = {
        "feature_categories": FEATURE_EXPLANATIONS,
        "node_features_ground_truth": NODE_FEATURES_GROUND_TRUTH,
        "node_features_partial_observation": NODE_FEATURES_PARTIAL_OBS,
        "edge_features": EDGE_FEATURES,
        "state_features": STATE_FEATURES,
        "action_features": ACTION_FEATURES,
        "action_blocks": ACTION_BLOCKS,
        "state_blocks": STATE_BLOCKS,
        "node_blocks_ground_truth": NODE_BLOCKS_GROUND_TRUTH,
        "node_blocks_partial_observation": NODE_BLOCKS_PARTIAL_OBS,
        "variants": [v.to_dict() for v in variants],
    }
    _write_json(path, payload)


def _worker_main() -> int:
    parser = argparse.ArgumentParser(description="Internal worker mode for policy feature ablations.")
    parser.add_argument("--project_root", required=True, type=str)
    parser.add_argument("--variant_json", required=True, type=str)
    parser.add_argument("--train_out_dir", required=True, type=str)
    parser.add_argument("--eval_out_dir", required=True, type=str)
    parser.add_argument("--worker_result_json", required=True, type=str)
    parser.add_argument("--train_cli_json", required=True, type=str)
    parser.add_argument("--eval_cli_json", required=True, type=str)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    variant_payload = _read_json(Path(args.variant_json).resolve())
    train_cli = json.loads(Path(args.train_cli_json).read_text(encoding="utf-8"))
    eval_cli = json.loads(Path(args.eval_cli_json).read_text(encoding="utf-8"))

    if not isinstance(train_cli, list) or not all(isinstance(x, str) for x in train_cli):
        raise ValueError("train_cli_json must contain a JSON string list.")
    if not isinstance(eval_cli, list) or not all(isinstance(x, str) for x in eval_cli):
        raise ValueError("eval_cli_json must contain a JSON string list.")

    sys.path.insert(0, str(project_root))

    import torch  # type: ignore
    import temporal_graph_dataset as tgd  # type: ignore

    keep_node_gt = list(variant_payload.get("keep_node_gt", []))
    keep_node_po = list(variant_payload.get("keep_node_po", []))
    keep_edge = list(variant_payload.get("keep_edge", []))
    keep_state = list(variant_payload.get("keep_state", []))
    keep_action = list(variant_payload.get("keep_action", []))

    node_idx_gt = [NODE_FEATURES_GROUND_TRUTH.index(x) for x in keep_node_gt]
    node_idx_po = [NODE_FEATURES_PARTIAL_OBS.index(x) for x in keep_node_po]
    edge_idx = [EDGE_FEATURES.index(x) for x in keep_edge]
    state_idx = [STATE_FEATURES.index(x) for x in keep_state]
    action_idx = [ACTION_FEATURES.index(x) for x in keep_action]

    orig_load_graph = tgd.TemporalGraphDataset._load_graph
    orig_state_builder = tgd._build_state_summary_features_from_graph
    orig_action_builder = tgd._build_action_features_from_manifest_row

    def patched_load_graph(self, file_key: str):
        data = orig_load_graph(self, file_key)
        if hasattr(data, "clone"):
            data = data.clone()

        x = getattr(data, "x", None)
        if torch.is_tensor(x) and x.dim() == 2:
            data._ablation_original_x = x.clone()
            feat_dim = int(x.size(1))
            if feat_dim >= 22:
                idx = node_idx_gt
            elif feat_dim >= 16:
                idx = node_idx_po
            else:
                idx = list(range(feat_dim))
            if idx:
                data.x = x[:, idx].contiguous()
            else:
                data.x = x.new_zeros((int(x.size(0)), 0))

        edge_attr = getattr(data, "edge_attr", None)
        if torch.is_tensor(edge_attr):
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            data._ablation_original_edge_attr = edge_attr.clone()
            if edge_idx:
                valid_idx = [i for i in edge_idx if i < int(edge_attr.size(1))]
                if valid_idx:
                    data.edge_attr = edge_attr[:, valid_idx].contiguous()
                else:
                    data.edge_attr = edge_attr.new_zeros((int(edge_attr.size(0)), 0))
            else:
                data.edge_attr = edge_attr.new_zeros((int(edge_attr.size(0)), 0))
        return data

    def patched_state_builder(data, calibration=None):
        original_x = getattr(data, "_ablation_original_x", None)
        original_edge_attr = getattr(data, "_ablation_original_edge_attr", None)

        had_x = hasattr(data, "x")
        had_edge_attr = hasattr(data, "edge_attr")
        current_x = getattr(data, "x", None)
        current_edge_attr = getattr(data, "edge_attr", None)

        if torch.is_tensor(original_x):
            data.x = original_x
        if torch.is_tensor(original_edge_attr):
            data.edge_attr = original_edge_attr

        try:
            full = orig_state_builder(data, calibration=calibration)
        finally:
            if had_x:
                data.x = current_x
            elif hasattr(data, "x"):
                delattr(data, "x")
            if had_edge_attr:
                data.edge_attr = current_edge_attr
            elif hasattr(data, "edge_attr"):
                delattr(data, "edge_attr")

        full = full.to(torch.float32).view(-1)
        if state_idx:
            return full[state_idx].contiguous()
        return full.new_zeros((0,), dtype=torch.float32)

    def patched_action_builder(row: Dict[str, Any]):
        full = orig_action_builder(row).to(torch.float32).view(-1)
        if action_idx:
            return full[action_idx].contiguous()
        return full.new_zeros((0,), dtype=torch.float32)

    tgd.TemporalGraphDataset._load_graph = patched_load_graph
    tgd._build_state_summary_features_from_graph = patched_state_builder
    tgd._build_action_features_from_manifest_row = patched_action_builder

    import train_amr_dygformer  # type: ignore
    import evaluate_policy_selector  # type: ignore

    train_exit = 0
    eval_exit = 0
    train_error = ""
    eval_error = ""
    summary: Dict[str, Any] = {}

    try:
        sys.argv = ["train_amr_dygformer.py"] + train_cli
        train_amr_dygformer.main()
    except SystemExit as exc:
        train_exit = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:  # pragma: no cover
        train_exit = 1
        train_error = f"{type(exc).__name__}: {exc}"

    if train_exit == 0:
        try:
            sys.argv = ["evaluate_policy_selector.py"] + eval_cli
            eval_result = evaluate_policy_selector.main()
            if isinstance(eval_result, int):
                eval_exit = int(eval_result)
        except SystemExit as exc:
            eval_exit = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:  # pragma: no cover
            eval_exit = 1
            eval_error = f"{type(exc).__name__}: {exc}"

    eval_summary_path = Path(args.eval_out_dir) / "policy_selection_summary.json"
    if train_exit == 0 and eval_exit == 0 and eval_summary_path.exists():
        summary = _read_json(eval_summary_path)

    payload = {
        "train_exit": int(train_exit),
        "eval_exit": int(eval_exit),
        "train_error": train_error,
        "eval_error": eval_error,
        "eval_summary": summary,
    }
    _write_json(Path(args.worker_result_json), payload)
    return 0 if (train_exit == 0 and eval_exit == 0) else 1


def _master_main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone feature ablations for the AMR causal policy selector.")
    parser.add_argument("--project_root", required=True, type=Path, help="Root directory of the AMR codebase.")
    parser.add_argument("--data_folder", required=True, type=Path, help="Causal policy dataset root containing policy_manifest.csv.")
    parser.add_argument("--out_root", required=True, type=Path, help="Output directory for ablation artifacts.")
    parser.add_argument("--task", required=True, type=str, help="Training/evaluation task name, e.g. oracle_best_action_h14.")
    parser.add_argument("--oracle_metric", default="auto", type=str, help="Oracle metric passed to evaluate_policy_selector.py.")
    parser.add_argument("--selection_direction", default="auto", type=str, choices=["auto", "minimize", "maximize"])
    parser.add_argument("--candidate_interventions_json", default="", type=Path, help="Optional candidate_interventions.json for evaluation labels.")
    parser.add_argument("--train_extra", default="", type=str, help="Additional raw CLI args appended to train_amr_dygformer.py.")
    parser.add_argument("--eval_extra", default="", type=str, help="Additional raw CLI args appended to evaluate_policy_selector.py.")
    parser.add_argument("--eval_split", default="test", type=str, choices=["train", "validation", "val", "test"])
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of built-in variant names.")
    parser.add_argument("--primary_metric", default="policy_accuracy", type=str, choices=["policy_accuracy", "strict_policy_accuracy", "top2_accuracy", "strict_top2_accuracy", "macro_f1", "weighted_f1", "accuracy"], help="Metric used to choose the minimal acceptable architecture.")
    parser.add_argument("--tolerance", default=0.01, type=float, help="Allowable drop from the best primary metric when choosing the smallest acceptable architecture.")
    parser.add_argument("--python_executable", default=sys.executable, type=str, help="Python executable to use for worker processes.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately if any ablation worker fails.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_folder = args.data_folder.resolve()
    out_root = args.out_root.resolve()
    _ensure_dir(out_root)

    variants = _resolve_variants(args.variants)
    train_T, train_sliding_step = _resolve_train_window_params(args.train_extra)
    state_mode = _discover_state_mode(project_root, data_folder, T=train_T, sliding_step=train_sliding_step)

    _write_feature_catalog(out_root / "feature_catalog.json", variants)

    rows: List[Dict[str, Any]] = []
    best_primary = -float("inf")

    for idx, variant in enumerate(variants, start=1):
        variant_dir = out_root / f"{idx:02d}_{variant.name}"
        train_dir = variant_dir / "train"
        eval_dir = variant_dir / "eval"
        meta_dir = variant_dir / "meta"
        _ensure_dir(train_dir)
        _ensure_dir(eval_dir)
        _ensure_dir(meta_dir)

        variant_json = meta_dir / "variant.json"
        train_cli_json = meta_dir / "train_cli.json"
        eval_cli_json = meta_dir / "eval_cli.json"
        worker_result_json = meta_dir / "worker_result.json"

        variant_payload = variant.to_dict()
        _write_json(variant_json, variant_payload)

        train_cli = _build_train_cli_args(args, variant, train_dir)
        eval_cli = _build_eval_cli_args(args, variant, train_dir, eval_dir)
        train_cli_json.write_text(json.dumps(train_cli, indent=2), encoding="utf-8")
        eval_cli_json.write_text(json.dumps(eval_cli, indent=2), encoding="utf-8")

        cmd = [
            str(args.python_executable),
            str(Path(__file__).resolve()),
            "--worker",
            "--project_root", str(project_root),
            "--variant_json", str(variant_json),
            "--train_out_dir", str(train_dir),
            "--eval_out_dir", str(eval_dir),
            "--worker_result_json", str(worker_result_json),
            "--train_cli_json", str(train_cli_json),
            "--eval_cli_json", str(eval_cli_json),
        ]

        print(f"[{idx}/{len(variants)}] Running variant '{variant.name}'", flush=True)
        completed = subprocess.run(cmd, check=False)

        worker_payload = _read_json(worker_result_json) if worker_result_json.exists() else {}
        eval_summary = worker_payload.get("eval_summary", {}) if isinstance(worker_payload, dict) else {}
        if not isinstance(eval_summary, dict):
            eval_summary = {}

        classification_metrics = eval_summary.get("classification_metrics", {})
        if not isinstance(classification_metrics, dict):
            classification_metrics = {}

        dims = _count_dims(variant)
        primary_metric_value = float("nan")
        if args.primary_metric in eval_summary:
            primary_metric_value = _safe_float(eval_summary.get(args.primary_metric))
        elif args.primary_metric in classification_metrics:
            primary_metric_value = _safe_float(classification_metrics.get(args.primary_metric))

        row: Dict[str, Any] = {
            "variant": variant.name,
            "description": variant.description,
            "status": "ok" if completed.returncode == 0 else "failed",
            "worker_returncode": int(completed.returncode),
            "train_exit": int(worker_payload.get("train_exit", 1)) if isinstance(worker_payload, dict) else 1,
            "eval_exit": int(worker_payload.get("eval_exit", 1)) if isinstance(worker_payload, dict) else 1,
            "train_error": str(worker_payload.get("train_error", "")) if isinstance(worker_payload, dict) else "",
            "eval_error": str(worker_payload.get("eval_error", "")) if isinstance(worker_payload, dict) else "",
            "n_states": int(eval_summary.get("n_states", 0)) if eval_summary else 0,
            "policy_accuracy": _safe_float(eval_summary.get("policy_accuracy")),
            "strict_policy_accuracy": _safe_float(eval_summary.get("strict_policy_accuracy")),
            "top2_accuracy": _safe_float(eval_summary.get("top2_accuracy")),
            "strict_top2_accuracy": _safe_float(eval_summary.get("strict_top2_accuracy")),
            "mean_regret": _safe_float((eval_summary.get("regret", {}) or {}).get("mean")),
            "median_regret": _safe_float((eval_summary.get("regret", {}) or {}).get("median")),
            "baseline_improvement_rate": _safe_float(eval_summary.get("baseline_improvement_rate")),
            "tie_state_rate": _safe_float(eval_summary.get("tie_state_rate")),
            "macro_f1": _safe_float(classification_metrics.get("macro_f1")),
            "weighted_f1": _safe_float(classification_metrics.get("weighted_f1")),
            "accuracy": _safe_float(classification_metrics.get("accuracy")),
            "primary_metric_name": args.primary_metric,
            "primary_metric_value": primary_metric_value,
            "feature_keep_summary": _render_feature_keep_description(variant, state_mode),
            "kept_node_features": "|".join(list(variant.keep_node_gt) if state_mode == "ground_truth" else list(variant.keep_node_po)),
            "kept_edge_features": "|".join(list(variant.keep_edge)),
            "kept_state_features": "|".join(list(variant.keep_state)),
            "kept_action_features": "|".join(list(variant.keep_action) if variant.use_action_conditioning else []),
            "use_action_conditioning": bool(variant.use_action_conditioning),
            **dims,
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
        }
        rows.append(row)

        if math.isfinite(primary_metric_value):
            best_primary = max(best_primary, primary_metric_value)

        if completed.returncode != 0 and args.stop_on_error:
            break

    if not rows:
        raise RuntimeError("No ablation rows were generated.")

    acceptable_rows: List[Dict[str, Any]] = []
    for row in rows:
        pm = _safe_float(row.get("primary_metric_value"))
        if row.get("status") == "ok" and math.isfinite(pm) and pm >= (best_primary - float(args.tolerance)):
            acceptable_rows.append(row)

    def _variant_total_dim(row: Dict[str, Any]) -> int:
        if state_mode == "ground_truth":
            return int(row.get("total_dim_gt", 0))
        return int(row.get("total_dim_po", 0))

    selected_name = ""
    if acceptable_rows:
        acceptable_rows.sort(
            key=lambda r: (
                _variant_total_dim(r),
                -_safe_float(r.get("primary_metric_value"), default=-1e9),
                _safe_float(r.get("mean_regret"), default=1e9),
            )
        )
        selected_name = str(acceptable_rows[0].get("variant", ""))

    for row in rows:
        row["is_selected_minimal"] = bool(str(row.get("variant", "")) == selected_name)
        row["best_primary_metric_value"] = best_primary
        row["primary_tolerance"] = float(args.tolerance)
        row["within_tolerance_of_best"] = bool(
            row.get("status") == "ok" and math.isfinite(_safe_float(row.get("primary_metric_value"))) and _safe_float(row.get("primary_metric_value")) >= (best_primary - float(args.tolerance))
        )

    csv_path = out_root / "ablation_results.csv"
    json_path = out_root / "ablation_results.json"
    tex_path = out_root / "ablation_results_table.tex"
    summary_txt_path = out_root / "ablation_summary.txt"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_json(json_path, {"rows": rows, "state_mode": state_mode, "primary_metric": args.primary_metric, "best_primary_metric_value": best_primary, "selected_minimal_variant": selected_name})

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("status") != "ok",
            -_safe_float(r.get("primary_metric_value"), default=-1e9),
            _safe_float(r.get("mean_regret"), default=1e9),
            _variant_total_dim(r),
        ),
    )
    tex_path.write_text(
        _generate_latex_table(sorted_rows, state_mode=state_mode, primary_metric=args.primary_metric, tolerance=float(args.tolerance)),
        encoding="utf-8",
    )

    lines = [
        f"State mode inferred from dataset: {state_mode}",
        f"Primary metric: {args.primary_metric}",
        f"Best observed primary metric: {best_primary:.6f}" if math.isfinite(best_primary) else "Best observed primary metric: NaN",
    ]
    if selected_name:
        selected_row = next((r for r in rows if str(r.get("variant", "")) == selected_name), None)
        total_dim = _variant_total_dim(selected_row) if selected_row else -1
        lines.extend(
            [
                f"Selected minimal variant within tolerance {args.tolerance:.6f}: {selected_name}",
                f"Total tested dimension count for selected variant: {total_dim}",
                f"Feature split for selected variant: {selected_row.get('feature_keep_summary', '') if selected_row else ''}",
            ]
        )
    else:
        lines.append("No variant completed successfully within the specified tolerance.")
    summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Ablation study complete.", flush=True)
    print(f"Results CSV: {csv_path}", flush=True)
    print(f"Results JSON: {json_path}", flush=True)
    print(f"LaTeX table: {tex_path}", flush=True)
    print(f"Summary: {summary_txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        filtered = [arg for arg in sys.argv[1:] if arg != "--worker"]
        sys.argv = [sys.argv[0]] + filtered
        raise SystemExit(_worker_main())
    raise SystemExit(_master_main())
