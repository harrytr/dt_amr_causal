
#!/usr/bin/env python3
"""
train_simple_policy_baseline.py

Train a simple baseline classifier to predict the oracle-best action directly
from explicit state features derived from the policy manifest window.

Purpose
-------
This script is designed as a diagnostic baseline for the causal AMR policy
selection task. It ignores the action-conditioned GNN and instead asks:

    "Can a simple model predict which intervention is oracle-best for a state?"

It builds one sample per unique state_id by:
1. Reading policy_manifest.csv
2. Grouping rows by state_id
3. Computing explicit state features from the PT graph window
4. Computing the oracle-best action from a chosen oracle metric
5. Training a simple classifier such as logistic regression or random forest

By default, the state features are derived only from:
- g.x
- g.edge_index
- g.edge_attr

for each graph in the temporal window, then aggregated across time.
This avoids leaking latent simulator-only fields.

Example
-------
python train_simple_policy_baseline.py \
    --manifest_csv experiments_causal_results/TRACK_ground_truth/work/causal_policy_monthly_shock_superspreader/policy_manifest.csv \
    --oracle_metric y_h14_trans_import_res_gain \
    --model logistic \
    --out_dir baseline_policy_classifier

Outputs
-------
- metrics_summary.json
- per_state_predictions.csv
- confusion_matrix.png
- feature_importance.csv (for tree models or linear coefficients)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        try:
            if hasattr(x, "item"):
                return float(x.item())
        except Exception:
            return float(default)
    return float(default)


def _safe_json_loads(s: Any, default: Any) -> Any:
    try:
        return json.loads(str(s))
    except Exception:
        return default


def _normalize_count(value: float, denom: float) -> float:
    return float(value) / float(max(1.0, denom))


def _load_graph(path: str):
    return torch.load(path, weights_only=False)


def _compute_graph_x_edge_features(g) -> Dict[str, float]:
    """
    Build explicit graph summary features from the tensors the model actually sees:
      - x
      - edge_index
      - edge_attr

    This intentionally avoids latent simulator attributes such as graph-level
    future totals or raw AMR-state metadata not present in x.
    """
    features: Dict[str, float] = {}

    x = getattr(g, "x", None)
    edge_index = getattr(g, "edge_index", None)
    edge_attr = getattr(g, "edge_attr", None)

    if not torch.is_tensor(x) or x.ndim != 2 or x.size(0) == 0:
        num_nodes = int(getattr(g, "num_nodes", 0))
        num_edges = int(edge_index.size(1)) if torch.is_tensor(edge_index) and edge_index.ndim == 2 else 0
        features["num_nodes_scaled"] = min(float(num_nodes) / 500.0, 1.0)
        features["num_edges_scaled"] = min(float(num_edges) / 5000.0, 1.0)
        features["density"] = float(num_edges) / float(max(1, num_nodes * num_nodes))
        return features

    x = x.to(torch.float32)
    num_nodes = int(x.size(0))
    num_feats = int(x.size(1))
    num_edges = int(edge_index.size(1)) if torch.is_tensor(edge_index) and edge_index.ndim == 2 else 0

    features["num_nodes_scaled"] = min(float(num_nodes) / 500.0, 1.0)
    features["num_edges_scaled"] = min(float(num_edges) / 5000.0, 1.0)
    features["density"] = float(num_edges) / float(max(1, num_nodes * num_nodes))

    # Mean and std for all x columns
    for j in range(num_feats):
        col = x[:, j]
        features[f"x_mean_{j}"] = float(col.mean().item())
        if col.numel() > 1:
            features[f"x_std_{j}"] = float(col.std(unbiased=False).item())
        else:
            features[f"x_std_{j}"] = 0.0

    # Interpretable summaries if expected columns exist
    role_col = x[:, 0] if num_feats >= 1 else None
    state_col = x[:, 1] if num_feats >= 2 else None
    abx_col = x[:, 2] if num_feats >= 3 else None
    iso_col = x[:, 3] if num_feats >= 4 else None
    new_cr_col = x[:, 4] if num_feats >= 5 else None
    new_ir_col = x[:, 5] if num_feats >= 6 else None

    if role_col is not None:
        staff_count = float((role_col >= 0.5).sum().item())
        patient_count = float(num_nodes) - staff_count
        features["staff_frac"] = _normalize_count(staff_count, float(num_nodes))
        features["patient_frac"] = _normalize_count(patient_count, float(num_nodes))

    if state_col is not None:
        # Works for both tracks because the meaning of x[:,1] comes from the track.
        rounded = torch.round(state_col).to(torch.long)
        unique_vals = set(int(v) for v in rounded.unique().cpu().tolist())
        if unique_vals.issubset({0, 1, 2, 3, 4}):
            for v in range(5):
                features[f"state_frac_{v}"] = _normalize_count(float((rounded == v).sum().item()), float(num_nodes))
        else:
            # For binary / continuous observed-style channels
            features["state_positive_frac"] = _normalize_count(float((state_col > 0.5).sum().item()), float(num_nodes))
        features["state_min"] = float(state_col.min().item())
        features["state_max"] = float(state_col.max().item())
        features["state_range"] = float(features["state_max"] - features["state_min"])

    if abx_col is not None:
        features["abx_positive_frac"] = _normalize_count(float((abx_col > 0.0).sum().item()), float(num_nodes))

    if iso_col is not None:
        features["iso_positive_frac"] = _normalize_count(float((iso_col > 0.5).sum().item()), float(num_nodes))

    if new_cr_col is not None:
        features["new_cr_sum_norm"] = _normalize_count(float(new_cr_col.sum().item()), float(num_nodes))
        features["new_cr_positive_frac"] = _normalize_count(float((new_cr_col > 0.0).sum().item()), float(num_nodes))

    if new_ir_col is not None:
        features["new_ir_sum_norm"] = _normalize_count(float(new_ir_col.sum().item()), float(num_nodes))
        features["new_ir_positive_frac"] = _normalize_count(float((new_ir_col > 0.0).sum().item()), float(num_nodes))

    # Edge features
    if torch.is_tensor(edge_attr) and edge_attr.numel() > 0:
        ea = edge_attr.to(torch.float32)
        if ea.ndim == 1:
            ea = ea.view(-1, 1)
        if ea.size(1) >= 1:
            ew = ea[:, 0]
            features["edge_weight_mean"] = float(ew.mean().item())
            if ew.numel() > 1:
                features["edge_weight_std"] = float(ew.std(unbiased=False).item())
            else:
                features["edge_weight_std"] = 0.0
        if ea.size(1) >= 2:
            et = ea[:, 1]
            features["edge_type_mean"] = float(et.mean().item())
            for t in range(4):
                features[f"edge_type_frac_{t}"] = _normalize_count(float((torch.round(et).to(torch.long) == t).sum().item()), float(et.numel()))
    else:
        features["edge_weight_mean"] = 0.0
        features["edge_weight_std"] = 0.0
        features["edge_type_mean"] = 0.0

    # Degree summaries
    if torch.is_tensor(edge_index) and edge_index.ndim == 2 and edge_index.size(1) > 0:
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        out_deg = torch.bincount(src, minlength=num_nodes).to(torch.float32)
        in_deg = torch.bincount(dst, minlength=num_nodes).to(torch.float32)
        features["out_deg_mean"] = float(out_deg.mean().item())
        features["in_deg_mean"] = float(in_deg.mean().item())
        features["out_deg_max_norm"] = _normalize_count(float(out_deg.max().item()), float(num_nodes))
        features["in_deg_max_norm"] = _normalize_count(float(in_deg.max().item()), float(num_nodes))
    else:
        features["out_deg_mean"] = 0.0
        features["in_deg_mean"] = 0.0
        features["out_deg_max_norm"] = 0.0
        features["in_deg_max_norm"] = 0.0

    return features


def _aggregate_window_feature_dicts(per_day_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not per_day_features:
        return {}

    keys = sorted({k for d in per_day_features for k in d.keys()})
    out: Dict[str, float] = {}

    for key in keys:
        vals = np.asarray([float(d.get(key, 0.0)) for d in per_day_features], dtype=float)
        out[f"{key}_last"] = float(vals[-1])
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
        out[f"{key}_trend"] = float(vals[-1] - vals[0])

    return out


def _extract_window_state_features(window_pt_paths: Sequence[str]) -> Dict[str, float]:
    per_day: List[Dict[str, float]] = []
    for p in window_pt_paths:
        g = _load_graph(str(p))
        per_day.append(_compute_graph_x_edge_features(g))
    return _aggregate_window_feature_dicts(per_day)


def _read_manifest(manifest_csv: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_csv}")
    return rows


def _resolve_window_paths(row: Dict[str, Any]) -> List[str]:
    raw = _safe_json_loads(row.get("window_pt_json", ""), [])
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Invalid or empty window_pt_json for state_id={row.get('state_id', '')}")
    paths = [os.path.abspath(str(p)) for p in raw]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing PT path from window_pt_json: {p}")
    return paths


def _group_rows_by_state(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        state_id = str(row.get("state_id", "")).strip()
        if state_id == "":
            raise ValueError("Manifest row missing state_id.")
        grouped[state_id].append(row)
    return grouped


def _pick_oracle_best_action(
    state_rows: Sequence[Dict[str, Any]],
    oracle_metric: str,
    selection_direction: str,
) -> Tuple[str, str, float]:
    direction = str(selection_direction).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("selection_direction must be 'maximize' or 'minimize'.")

    best_row: Optional[Dict[str, Any]] = None
    best_value: Optional[float] = None

    for row in state_rows:
        if oracle_metric not in row:
            raise KeyError(f"Oracle metric '{oracle_metric}' missing from manifest row.")
        val = _safe_float(row.get(oracle_metric), default=float("nan"))
        if not math.isfinite(val):
            continue

        if best_value is None:
            best_value = val
            best_row = row
            continue

        if direction == "maximize":
            if val > best_value:
                best_value = val
                best_row = row
        else:
            if val < best_value:
                best_value = val
                best_row = row

    if best_row is None or best_value is None:
        raise ValueError(f"No finite oracle values found for metric '{oracle_metric}'.")

    action_id = str(best_row.get("action_id", "")).strip()
    action_name = str(best_row.get("action_name", action_id)).strip()
    return action_id, action_name, float(best_value)


def build_state_level_dataset(
    manifest_csv: str,
    oracle_metric: str,
    selection_direction: str,
    include_metadata_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[Dict[str, Any]], List[str]]:
    rows = _read_manifest(manifest_csv)
    grouped = _group_rows_by_state(rows)

    feature_rows: List[Dict[str, float]] = []
    targets: List[str] = []
    state_ids: List[str] = []
    splits: List[str] = []
    metadata_rows: List[Dict[str, Any]] = []

    for state_id, state_rows in sorted(grouped.items()):
        base_row = state_rows[0]
        window_paths = _resolve_window_paths(base_row)
        feats = _extract_window_state_features(window_paths)

        if include_metadata_features:
            feats["decision_day_scaled"] = min(_safe_float(base_row.get("decision_day", 0.0)) / 365.0, 1.0)
            feats["window_T_scaled"] = min(_safe_float(base_row.get("window_T", len(window_paths))) / 64.0, 1.0)
            feats["seed_scaled"] = _safe_float(base_row.get("seed", 0.0)) / 10000.0

        oracle_action_id, oracle_action_name, oracle_value = _pick_oracle_best_action(
            state_rows=state_rows,
            oracle_metric=oracle_metric,
            selection_direction=selection_direction,
        )

        feature_rows.append(feats)
        targets.append(oracle_action_id)
        state_ids.append(state_id)
        splits.append(str(base_row.get("split", "")).strip().lower())
        metadata_rows.append(
            {
                "state_id": state_id,
                "split": str(base_row.get("split", "")).strip().lower(),
                "oracle_best_action_id": oracle_action_id,
                "oracle_best_action_display_name": oracle_action_name,
                "oracle_best_value": oracle_value,
                "decision_day": _safe_float(base_row.get("decision_day", 0.0)),
            }
        )

    feature_names = sorted({k for row in feature_rows for k in row.keys()})
    X = np.zeros((len(feature_rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(feature_rows):
        for j, name in enumerate(feature_names):
            X[i, j] = float(row.get(name, 0.0))

    y = np.asarray(targets, dtype=object)
    return X, y, state_ids, splits, metadata_rows, feature_names


def _make_model(model_name: str, random_state: int):
    model_name = str(model_name).strip().lower()
    if model_name == "logistic":
        clf = LogisticRegression(
            multi_class="multinomial",
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    raise ValueError("Unsupported model. Use 'logistic' or 'random_forest'.")


def _evaluate_split(
    model,
    X: np.ndarray,
    y_true_text: np.ndarray,
    encoder: LabelEncoder,
    split_name: str,
) -> Dict[str, Any]:
    y_true = encoder.transform(y_true_text)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    metrics: Dict[str, Any] = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top2_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=min(2, y_prob.shape[1]), labels=np.arange(y_prob.shape[1]))),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[str(x) for x in encoder.classes_],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(encoder.classes_))).tolist(),
        "classes": [str(x) for x in encoder.classes_],
    }
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: str, title: str) -> None:
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.imshow(norm_cm, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion", fontsize=11)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = norm_cm[i, j] * 100.0
            txt = f"{int(cm[i, j])}\n{pct:.1f}%"
            color = "white" if norm_cm[i, j] >= 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance(
    model,
    feature_names: Sequence[str],
    out_csv: str,
) -> None:
    rows: List[Tuple[str, float]] = []
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
        rows = [(str(name), float(val)) for name, val in zip(feature_names, importances)]
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        if coef.ndim == 2:
            imp = np.mean(np.abs(coef), axis=0)
        else:
            imp = np.abs(coef)
        rows = [(str(name), float(val)) for name, val in zip(feature_names, imp)]

    if not rows:
        return

    rows.sort(key=lambda x: (-x[1], x[0]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        for name, val in rows:
            writer.writerow([name, f"{val:.10f}"])


def _save_predictions_csv(
    model,
    X: np.ndarray,
    y_text: np.ndarray,
    state_ids: Sequence[str],
    splits: Sequence[str],
    metadata_rows: Sequence[Dict[str, Any]],
    encoder: LabelEncoder,
    out_csv: str,
) -> None:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    classes = [str(x) for x in encoder.classes_]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "state_id",
            "split",
            "oracle_best_action_id",
            "predicted_action_id",
            "policy_match",
            "oracle_best_action_display_name",
            "oracle_best_value",
            "decision_day",
        ] + [f"prob_{cls}" for cls in classes]
        writer.writerow(header)

        for i in range(len(state_ids)):
            meta = metadata_rows[i]
            pred_text = str(classes[int(y_pred[i])])
            oracle_text = str(y_text[i])
            row = [
                str(state_ids[i]),
                str(splits[i]),
                oracle_text,
                pred_text,
                int(pred_text == oracle_text),
                str(meta.get("oracle_best_action_display_name", oracle_text)),
                float(meta.get("oracle_best_value", 0.0)),
                float(meta.get("decision_day", 0.0)),
            ] + [float(p) for p in y_prob[i].tolist()]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--oracle_metric", type=str, required=True)
    parser.add_argument("--selection_direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "random_forest"])
    parser.add_argument("--out_dir", type=str, default="baseline_policy_classifier")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--include_metadata_features", type=str, default="true", choices=["true", "false"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_metadata_features = str(args.include_metadata_features).strip().lower() == "true"

    X, y_text, state_ids, splits, metadata_rows, feature_names = build_state_level_dataset(
        manifest_csv=os.path.abspath(args.manifest_csv),
        oracle_metric=str(args.oracle_metric),
        selection_direction=str(args.selection_direction),
        include_metadata_features=include_metadata_features,
    )

    splits_arr = np.asarray(splits, dtype=object)
    train_mask = splits_arr == "train"
    val_mask = np.isin(splits_arr, ["val", "validation"])
    test_mask = splits_arr == "test"

    if int(train_mask.sum()) == 0:
        raise ValueError("No train states found in manifest.")
    if int(val_mask.sum()) == 0 and int(test_mask.sum()) == 0:
        raise ValueError("No validation or test states found in manifest.")

    encoder = LabelEncoder()
    y_train_text = y_text[train_mask]
    encoder.fit(y_train_text)

    # Restrict evaluation to labels seen in training.
    known_mask = np.isin(y_text, encoder.classes_)
    if not np.all(known_mask):
        print(
            f"⚠️ Dropping {int((~known_mask).sum())} states whose oracle action was unseen in train.",
            flush=True,
        )

    train_mask = train_mask & known_mask
    val_mask = val_mask & known_mask
    test_mask = test_mask & known_mask

    X_train = X[train_mask]
    y_train = y_text[train_mask]

    model = _make_model(args.model, args.random_state)
    model.fit(X_train, encoder.transform(y_train))

    summary: Dict[str, Any] = {
        "manifest_csv": os.path.abspath(args.manifest_csv),
        "oracle_metric": str(args.oracle_metric),
        "selection_direction": str(args.selection_direction),
        "model": str(args.model),
        "n_states_total": int(len(y_text)),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "classes": [str(x) for x in encoder.classes_],
        "class_counts_train": dict(Counter([str(x) for x in y_train.tolist()])),
        "feature_count": int(len(feature_names)),
    }

    if int(val_mask.sum()) > 0:
        val_metrics = _evaluate_split(model, X[val_mask], y_text[val_mask], encoder, "validation")
        summary["validation"] = val_metrics
        _plot_confusion_matrix(
            cm=np.asarray(val_metrics["confusion_matrix"]),
            class_names=val_metrics["classes"],
            out_path=str(out_dir / "confusion_matrix_validation.png"),
            title="Simple baseline confusion matrix (validation)",
        )

    if int(test_mask.sum()) > 0:
        test_metrics = _evaluate_split(model, X[test_mask], y_text[test_mask], encoder, "test")
        summary["test"] = test_metrics
        _plot_confusion_matrix(
            cm=np.asarray(test_metrics["confusion_matrix"]),
            class_names=test_metrics["classes"],
            out_path=str(out_dir / "confusion_matrix_test.png"),
            title="Simple baseline confusion matrix (test)",
        )

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _save_feature_importance(
        model=model,
        feature_names=feature_names,
        out_csv=str(out_dir / "feature_importance.csv"),
    )

    _save_predictions_csv(
        model=model,
        X=X,
        y_text=y_text,
        state_ids=state_ids,
        splits=splits,
        metadata_rows=metadata_rows,
        encoder=encoder,
        out_csv=str(out_dir / "per_state_predictions.csv"),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
