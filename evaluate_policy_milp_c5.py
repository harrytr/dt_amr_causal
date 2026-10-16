#!/usr/bin/env python3
from __future__ import annotations

"""
evaluate_policy_milp_c5.py
==========================

STEP C5 add-on for the causal AMR policy pathway.

The original C1--C4 experiment is not modified. C5 consumes the saved causal
policy dataset, frozen C2 policy model, held-out C3 action scores, and C4 MILP
rule configuration to quantify cross-state score behaviour and constrained
allocation performance.

C5 addresses the two issues evaluated in the manuscript and Supplementary
Information:

1. Whether the original baseline-relative policy scores carry information that
   is comparable across different decision states.
2. How the original raw-score MILP behaves when budget, maximum-use, and
   cooldown restrictions are activated and its selected schedules are evaluated
   using simulator-derived held-out outcomes.

Reported analyses
-----------------
* Re-score the validation split using the frozen C2 model; no retraining occurs.
* Anchor each action score and simulator outcome to the baseline action within
  the same decision state.
* Measure cross-state ordering of the original raw baseline-relative scores.
* Fit one monotone isotonic score-to-gain mapping on validation rows only and
  evaluate the frozen mapping on the untouched test split.
* Stress-test the original raw-score MILP at budget fractions 0.25, 0.50, and
  1.00 with the configured intervention-family cooldown and a maximum of 50%
  non-baseline decisions per trajectory.
* Evaluate the raw-score MILP against baseline allocation and a finite-rollout
  simulator-reference MILP under the same constraints.
* Use the validation-mapped MILP and a per-state normalised-score MILP only as
  score-scale sensitivity analyses.
* Quantify realised gain, reference gap, constraint activity, feasibility,
  solver optimality, and uncertainty across independent test trajectories.
* Reproduce the original C4 raw-score MILP as an implementation check.


Important scope
---------------
The finite-rollout reference uses existing simulator-derived test outcomes and
is not a true expected-benefit optimum. C5 is a retrospective fixed-state
allocation benchmark and does not establish prospective adaptive control.

Outputs
-------
The script writes machine-readable CSV/JSON files, 600-dpi figures, LaTeX
Tables, figure captions, and a manuscript bundle for the two observation tracks.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix
    from scipy.stats import pearsonr, spearmanr
except Exception as exc:  # pragma: no cover - dependency failure is explicit at runtime
    raise RuntimeError("C5 requires scipy with scipy.optimize.milp and scipy.stats.") from exc

try:
    from sklearn.isotonic import IsotonicRegression
except Exception as exc:  # pragma: no cover
    raise RuntimeError("C5 requires scikit-learn for validation-fitted isotonic calibration.") from exc


EPS = 1e-12
TIE_BREAK_EPS = 1e-9


@dataclass(frozen=True)
class ScoreRow:
    state_id: str
    split: str
    seed: int
    decision_day: int
    action_id: str
    action_name: str
    action_display_name: str
    is_baseline: int
    pred_score: float
    oracle_value: float
    payload: Dict[str, Any]


@dataclass
class RelativeRow:
    state_id: str
    split: str
    seed: int
    decision_day: int
    action_id: str
    action_name: str
    action_display_name: str
    is_baseline: int
    pred_score: float
    baseline_pred_score: float
    raw_delta: float
    oracle_value: float
    baseline_oracle_value: float
    oracle_gain: float
    calibrated_gain: float = 0.0
    state_normalised_score: float = 0.0
    cost: float = 0.0
    cooldown_days: int = 0
    cooldown_group: str = ""
    config_max_uses: Optional[int] = None
    max_uses_group: str = ""
    rule_source: str = ""


class C5Error(RuntimeError):
    pass


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise C5Error(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise C5Error(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise C5Error(f"Missing CSV file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise C5Error("At least one budget fraction is required.")
    if any(value <= 0 for value in values):
        raise C5Error("Budget fractions must be positive.")
    return values


def _latex_escape(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _nature_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3)
    ax.grid(False)


def _savefig600(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()


_METHOD_LABELS = {
    "calibrated_milp": "validation-mapped MILP",
    "raw_score_milp": "raw-score MILP",
    "state_normalised_milp": "state-normalised MILP",
    "oracle_reference_milp": "finite-rollout reference MILP",
    "baseline": "baseline",
}


def _method_label(method: str) -> str:
    return _METHOD_LABELS.get(str(method), str(method).replace("_", " "))


def _load_score_rows(csv_path: Path, required_split: Optional[str] = None) -> List[ScoreRow]:
    raw_rows = _read_csv(csv_path)
    rows: List[ScoreRow] = []
    wanted = "" if required_split is None else str(required_split).strip().lower()
    if wanted == "val":
        wanted = "validation"

    for raw in raw_rows:
        split = str(raw.get("split", "")).strip().lower()
        if split == "val":
            split = "validation"
        if wanted and split != wanted:
            continue
        pred_score = _safe_float(raw.get("pred_score", float("nan")))
        oracle_value = _safe_float(raw.get("oracle_value", float("nan")))
        if not math.isfinite(pred_score) or not math.isfinite(oracle_value):
            continue
        rows.append(
            ScoreRow(
                state_id=str(raw.get("state_id", "")).strip(),
                split=split,
                seed=_safe_int(raw.get("seed", 0), 0),
                decision_day=_safe_int(raw.get("decision_day", 0), 0),
                action_id=str(raw.get("action_id", "")).strip(),
                action_name=str(raw.get("action_name", "")).strip(),
                action_display_name=str(raw.get("action_display_name", raw.get("action_name", ""))).strip(),
                is_baseline=_safe_int(raw.get("is_baseline", 0), 0),
                pred_score=pred_score,
                oracle_value=oracle_value,
                payload=dict(raw),
            )
        )

    if not rows:
        raise C5Error(f"No finite score/oracle rows found in {csv_path} for split={required_split!r}.")
    return _retain_complete_states(rows)


def _retain_complete_states(rows: Sequence[ScoreRow]) -> List[ScoreRow]:
    by_state: Dict[str, List[ScoreRow]] = defaultdict(list)
    for row in rows:
        if not row.state_id:
            raise C5Error("Encountered a score row with empty state_id.")
        by_state[row.state_id].append(row)

    action_sets = [tuple(sorted(row.action_id for row in group)) for group in by_state.values()]
    modal_action_set = Counter(action_sets).most_common(1)[0][0]
    kept: List[ScoreRow] = []
    dropped: List[str] = []
    for state_id, group in by_state.items():
        actions = tuple(sorted(row.action_id for row in group))
        baseline_count = sum(int(row.is_baseline) == 1 for row in group)
        if actions != modal_action_set or baseline_count != 1 or len(set(actions)) != len(actions):
            dropped.append(state_id)
            continue
        kept.extend(group)
    if not kept:
        raise C5Error("No complete states remained after action-set/baseline validation.")
    if dropped:
        print(f"C5: dropped {len(dropped)} incomplete/invalid states before analysis.", flush=True)
    return sorted(kept, key=lambda row: (row.seed, row.decision_day, row.state_id, row.action_id))


def _oracle_direction(eval_summary: Mapping[str, Any]) -> str:
    direction = str(eval_summary.get("oracle_direction", "")).strip().lower()
    if direction in {"maximize", "minimize"}:
        return direction
    metric = str(eval_summary.get("oracle_metric", "")).strip().lower()
    if "gain" in metric or "improvement" in metric or "benefit" in metric:
        return "maximize"
    return "minimize"


def _relative_rows(rows: Sequence[ScoreRow], oracle_direction: str) -> List[RelativeRow]:
    by_state: Dict[str, List[ScoreRow]] = defaultdict(list)
    for row in rows:
        by_state[row.state_id].append(row)

    relative: List[RelativeRow] = []
    for state_id, group in sorted(by_state.items(), key=lambda item: (item[1][0].seed, item[1][0].decision_day, item[0])):
        baseline = next((row for row in group if int(row.is_baseline) == 1), None)
        if baseline is None:
            raise C5Error(f"State {state_id} has no baseline action.")
        raw_deltas: List[float] = []
        temporary: List[RelativeRow] = []
        for row in sorted(group, key=lambda item: item.action_id):
            raw_delta = float(row.pred_score - baseline.pred_score)
            if oracle_direction == "maximize":
                oracle_gain = float(row.oracle_value - baseline.oracle_value)
            else:
                oracle_gain = float(baseline.oracle_value - row.oracle_value)
            relative_row = RelativeRow(
                state_id=row.state_id,
                split=row.split,
                seed=row.seed,
                decision_day=row.decision_day,
                action_id=row.action_id,
                action_name=row.action_name,
                action_display_name=row.action_display_name,
                is_baseline=row.is_baseline,
                pred_score=row.pred_score,
                baseline_pred_score=baseline.pred_score,
                raw_delta=raw_delta,
                oracle_value=row.oracle_value,
                baseline_oracle_value=baseline.oracle_value,
                oracle_gain=oracle_gain,
            )
            temporary.append(relative_row)
            if int(row.is_baseline) == 0:
                raw_deltas.append(abs(raw_delta))
        scale = max(raw_deltas) if raw_deltas else 0.0
        for row in temporary:
            row.state_normalised_score = 0.0 if scale <= EPS else float(row.raw_delta / scale)
            relative.append(row)
    return relative


def _ensure_validation_scores(
    evaluator_script: Path,
    data_folder: Path,
    trained_dir: Path,
    candidate_interventions_json: Path,
    out_dir: Path,
    split: str,
    oracle_metric: str,
    selection_direction: str,
    force: bool,
) -> Path:
    target = out_dir / "policy_action_scores_requested_split.csv"
    summary_path = out_dir / "policy_selection_summary.json"
    if target.is_file() and summary_path.is_file() and not force:
        return target

    cmd = [
        sys.executable,
        str(evaluator_script),
        "--data_folder", str(data_folder),
        "--trained_dir", str(trained_dir),
        "--split", str(split),
        "--oracle_metric", str(oracle_metric),
        "--selection_direction", str(selection_direction),
        "--emit_action_scores_csv", "true",
        "--metrics_only", "true",
        "--candidate_interventions_json", str(candidate_interventions_json),
        "--out_dir", str(out_dir),
    ]
    print("C5: scoring validation split with frozen C2 model...", flush=True)
    subprocess.run(cmd, check=True, cwd=str(evaluator_script.resolve().parent))
    if not target.is_file():
        raise C5Error(f"Validation scoring did not produce {target}")
    return target



def _fit_isotonic_calibrator(validation_rows: Sequence[RelativeRow]) -> Tuple[IsotonicRegression, float]:
    nonbaseline = [row for row in validation_rows if int(row.is_baseline) == 0]
    if len(nonbaseline) < 2:
        raise C5Error("At least two non-baseline validation rows are required for C5 calibration.")

    x = np.asarray([row.raw_delta for row in nonbaseline], dtype=float)
    y = np.asarray([row.oracle_gain for row in nonbaseline], dtype=float)

    # Add the baseline origin explicitly, then fit a monotone mapping on validation
    # data only. The fitted mapping is frozen before any test-set evaluation.
    x_iso = np.concatenate([x, np.asarray([0.0], dtype=float)])
    y_iso = np.concatenate([y, np.asarray([0.0], dtype=float)])
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    isotonic.fit(x_iso, y_iso)
    zero_anchor = float(isotonic.predict(np.asarray([0.0], dtype=float))[0])
    return isotonic, zero_anchor




def _apply_isotonic_calibrator(
    rows: Sequence[RelativeRow],
    isotonic: IsotonicRegression,
    zero_anchor: float,
) -> None:
    x = np.asarray([row.raw_delta for row in rows], dtype=float)
    mapped = isotonic.predict(x) - float(zero_anchor)
    for idx, row in enumerate(rows):
        row.calibrated_gain = float(mapped[idx])
        if int(row.is_baseline) == 1:
            row.calibrated_gain = 0.0



def _safe_corr(fn: Any, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    try:
        result = fn(x, y)
        value = result.statistic if hasattr(result, "statistic") else result[0]
        return float(value)
    except Exception:
        return float("nan")


def _calibration_line(pred: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
    if len(pred) < 2 or np.allclose(pred, pred[0]):
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(len(pred), dtype=float), pred])
    coeff, *_ = np.linalg.lstsq(design, obs, rcond=None)
    return float(coeff[0]), float(coeff[1])


def _cross_state_pairwise_accuracy(rows: Sequence[RelativeRow], value_attr: str) -> Dict[str, Any]:
    data = [row for row in rows if int(row.is_baseline) == 0]
    correct = 0
    total = 0
    pred_ties = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i].state_id == data[j].state_id:
                continue
            true_diff = float(data[i].oracle_gain - data[j].oracle_gain)
            if abs(true_diff) <= EPS:
                continue
            pred_diff = float(getattr(data[i], value_attr) - getattr(data[j], value_attr))
            total += 1
            if abs(pred_diff) <= EPS:
                pred_ties += 1
                continue
            if (pred_diff > 0) == (true_diff > 0):
                correct += 1
    return {
        "cross_state_pairwise_pairs": int(total),
        "cross_state_pairwise_accuracy": float(correct / total) if total else float("nan"),
        "cross_state_pairwise_predicted_tie_rate": float(pred_ties / total) if total else float("nan"),
    }


def _diagnostic_row(rows: Sequence[RelativeRow], attr: str, variant: str, subset: str) -> Dict[str, Any]:
    selected = [row for row in rows if subset == "all" or int(row.is_baseline) == 0]
    pred = np.asarray([float(getattr(row, attr)) for row in selected], dtype=float)
    obs = np.asarray([float(row.oracle_gain) for row in selected], dtype=float)
    residual = pred - obs
    intercept, slope = _calibration_line(pred, obs)
    pairwise = _cross_state_pairwise_accuracy(rows, attr)
    return {
        "variant": variant,
        "subset": subset,
        "n_rows": int(len(selected)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else float("nan"),
        "mae": float(np.mean(np.abs(residual))) if len(residual) else float("nan"),
        "bias_pred_minus_observed": float(np.mean(residual)) if len(residual) else float("nan"),
        "pearson_r": _safe_corr(pearsonr, pred, obs),
        "spearman_rho": _safe_corr(spearmanr, pred, obs),
        "calibration_intercept_observed_on_predicted": intercept,
        "calibration_slope_observed_on_predicted": slope,
        **pairwise,
    }



def _calibration_bins(rows: Sequence[RelativeRow], n_bins: int) -> List[Dict[str, Any]]:
    data = [row for row in rows if int(row.is_baseline) == 0]
    if not data:
        return []
    ordered = sorted(data, key=lambda row: (row.calibrated_gain, row.state_id, row.action_id))
    chunks = np.array_split(np.arange(len(ordered)), min(max(1, int(n_bins)), len(ordered)))
    out: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chosen = [ordered[int(i)] for i in chunk]
        pred = np.asarray([row.calibrated_gain for row in chosen], dtype=float)
        obs = np.asarray([row.oracle_gain for row in chosen], dtype=float)
        out.append({
            "bin": index,
            "n": len(chosen),
            "predicted_mean": float(pred.mean()),
            "observed_mean": float(obs.mean()),
            "predicted_min": float(pred.min()),
            "predicted_max": float(pred.max()),
        })
    return out




def _plot_calibration(rows: Sequence[RelativeRow], bins: Sequence[Mapping[str, Any]], path: Path) -> None:
    data = [row for row in rows if int(row.is_baseline) == 0]
    pred = np.asarray([row.calibrated_gain for row in data], dtype=float)
    obs = np.asarray([row.oracle_gain for row in data], dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, 5.3))
    ax.scatter(pred, obs, s=15, alpha=0.20, label="Test state-action rows")
    if bins:
        bx = np.asarray([float(row["predicted_mean"]) for row in bins])
        by = np.asarray([float(row["observed_mean"]) for row in bins])
        ax.plot(bx, by, "o-", linewidth=1.4, label="Binned observed gain")
    finite = np.concatenate([pred[np.isfinite(pred)], obs[np.isfinite(obs)]])
    if finite.size:
        lo = float(finite.min())
        hi = float(finite.max())
        if math.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="Identity")
    ax.set_xlabel("Validation-mapped predicted gain (resistant events)")
    ax.set_ylabel("Simulator-derived test gain (resistant events)")
    ax.set_title("Cross-state benefit calibration")
    ax.legend(frameon=False, fontsize=8)
    _nature_axes(ax)
    _savefig600(path)



def _resolve_action_rule(row: RelativeRow, config: Mapping[str, Any]) -> Dict[str, Any]:
    default_cost = _safe_float(config.get("default_action_cost", 0.0), 0.0)
    default_cooldown = max(0, _safe_int(config.get("default_cooldown_days", 0), 0))
    default_max_uses = config.get("default_max_uses", None)
    rules_by_id = config.get("action_rules_by_id", {}) or {}
    rules_by_name = config.get("action_rules_by_name", {}) or {}

    id_rule = rules_by_id.get(row.action_id, {}) if isinstance(rules_by_id, Mapping) else {}
    name_rule = rules_by_name.get(row.action_name, {}) if isinstance(rules_by_name, Mapping) else {}
    has_id = isinstance(id_rule, Mapping) and len(id_rule) > 0
    has_name = isinstance(name_rule, Mapping) and len(name_rule) > 0

    rule: Dict[str, Any] = {}
    if has_id:
        rule.update(dict(id_rule))
    if has_name:
        for key, value in dict(name_rule).items():
            rule.setdefault(key, value)

    default_group = row.action_id
    if not has_id and has_name and row.action_name:
        default_group = row.action_name
    explicit_group = str(rule.get("constraint_group", rule.get("rule_group", ""))).strip()
    constraint_group = explicit_group or default_group
    cooldown_group = str(rule.get("cooldown_group", constraint_group)).strip() or constraint_group
    max_uses_group = str(rule.get("max_uses_group", constraint_group)).strip() or constraint_group
    max_uses = rule.get("max_uses", default_max_uses)
    if max_uses is not None:
        max_uses = max(0, _safe_int(max_uses, 0))

    return {
        "cost": float(_safe_float(rule.get("cost", default_cost), default_cost)),
        "cooldown_days": max(0, _safe_int(rule.get("cooldown_days", default_cooldown), default_cooldown)),
        "cooldown_group": cooldown_group,
        "max_uses": max_uses,
        "max_uses_group": max_uses_group,
        "rule_source": "id" if has_id else ("name" if has_name else "default"),
    }


def _apply_rules(rows: Sequence[RelativeRow], config: Mapping[str, Any]) -> None:
    for row in rows:
        rule = _resolve_action_rule(row, config)
        row.cost = float(rule["cost"])
        row.cooldown_days = int(rule["cooldown_days"])
        row.cooldown_group = str(rule["cooldown_group"])
        row.config_max_uses = rule["max_uses"]
        row.max_uses_group = str(rule["max_uses_group"])
        row.rule_source = str(rule["rule_source"])


def _state_groups(rows: Sequence[RelativeRow]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row.state_id].append(index)
    return groups


def _cooldown_conflicts(rows: Sequence[RelativeRow]) -> List[Tuple[int, int]]:
    by_state = _state_groups(rows)
    state_meta = {
        state_id: rows[indices[0]]
        for state_id, indices in by_state.items()
    }
    ordered_states = sorted(by_state.keys(), key=lambda state_id: (state_meta[state_id].decision_day, state_id))
    pairs_by_state_group: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    cooldown_by_group: Dict[str, int] = {}
    for state_id, indices in by_state.items():
        for index in indices:
            row = rows[index]
            if int(row.is_baseline) == 1 or row.cooldown_days <= 0:
                continue
            group = row.cooldown_group or row.action_id
            pairs_by_state_group[(state_id, group)].append(index)
            cooldown_by_group[group] = max(cooldown_by_group.get(group, 0), int(row.cooldown_days))

    conflicts: set[Tuple[int, int]] = set()
    for group, cooldown_days in cooldown_by_group.items():
        for i, state_i in enumerate(ordered_states):
            indices_i = pairs_by_state_group.get((state_i, group), [])
            if not indices_i:
                continue
            day_i = state_meta[state_i].decision_day
            for state_j in ordered_states[i + 1:]:
                day_j = state_meta[state_j].decision_day
                if day_j - day_i > cooldown_days:
                    break
                indices_j = pairs_by_state_group.get((state_j, group), [])
                for one in indices_i:
                    for two in indices_j:
                        conflicts.add((min(one, two), max(one, two)))
    return sorted(conflicts)


def _config_max_use_groups(rows: Sequence[RelativeRow]) -> Dict[str, Tuple[int, List[int]]]:
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    grouped_limits: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.config_max_uses is None:
            continue
        group = row.max_uses_group or row.action_id
        grouped_indices[group].append(index)
        grouped_limits[group].append(int(row.config_max_uses))
    return {
        group: (min(grouped_limits[group]), grouped_indices[group])
        for group in grouped_indices
        if grouped_limits[group]
    }


def _allocation_feasibility(
    rows: Sequence[RelativeRow],
    selected: Sequence[int],
    budget: Optional[float],
    global_max_nonbaseline: Optional[int],
    use_cooldown: bool,
    use_config_max_uses: bool,
) -> Dict[str, Any]:
    by_state = _state_groups(rows)
    selected_set = set(int(index) for index in selected)
    one_per_state = all(sum(index in selected_set for index in indices) == 1 for indices in by_state.values())
    chosen = [rows[index] for index in selected]
    total_cost = float(sum(row.cost for row in chosen))
    nonbaseline_uses = int(sum(int(row.is_baseline) == 0 for row in chosen))
    cooldown_pairs = set(_cooldown_conflicts(rows)) if use_cooldown else set()
    cooldown_violations = int(sum((one in selected_set and two in selected_set) for one, two in cooldown_pairs))

    config_max_violations = 0
    if use_config_max_uses:
        for _, (limit, indices) in _config_max_use_groups(rows).items():
            uses = sum(index in selected_set for index in indices)
            config_max_violations += int(uses > limit)

    feasible = (
        one_per_state
        and (budget is None or total_cost <= float(budget) + 1e-8)
        and (global_max_nonbaseline is None or nonbaseline_uses <= int(global_max_nonbaseline))
        and cooldown_violations == 0
        and config_max_violations == 0
    )
    return {
        "feasible": bool(feasible),
        "one_action_per_state": bool(one_per_state),
        "total_cost": total_cost,
        "budget_slack": None if budget is None else float(budget - total_cost),
        "nonbaseline_uses": nonbaseline_uses,
        "global_max_nonbaseline": global_max_nonbaseline,
        "max_nonbaseline_slack": None if global_max_nonbaseline is None else int(global_max_nonbaseline - nonbaseline_uses),
        "cooldown_violations": cooldown_violations,
        "config_max_use_violations": int(config_max_violations),
    }


def _solve_milp(
    rows: Sequence[RelativeRow],
    utility: np.ndarray,
    budget: Optional[float],
    global_max_nonbaseline: Optional[int],
    use_cooldown: bool,
    use_config_max_uses: bool,
    time_limit_sec: float,
    mip_gap: float,
) -> Tuple[List[int], Dict[str, Any]]:
    n = len(rows)
    if n == 0 or len(utility) != n:
        raise C5Error("MILP received inconsistent rows/utilities.")
    if not np.isfinite(utility).all():
        raise C5Error("MILP utility contains non-finite values.")

    entries: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    lower: List[float] = []
    upper: List[float] = []

    def add_constraint(indices: Sequence[int], weights: Sequence[float], lo: float, hi: float) -> None:
        row_index = len(lower)
        entries.extend([row_index] * len(indices))
        columns.extend(int(index) for index in indices)
        values.extend(float(weight) for weight in weights)
        lower.append(float(lo))
        upper.append(float(hi))

    by_state = _state_groups(rows)
    for indices in by_state.values():
        add_constraint(indices, [1.0] * len(indices), 1.0, 1.0)

    if budget is not None:
        add_constraint(list(range(n)), [row.cost for row in rows], -np.inf, float(budget))

    if global_max_nonbaseline is not None:
        nonbaseline = [index for index, row in enumerate(rows) if int(row.is_baseline) == 0]
        if nonbaseline:
            add_constraint(nonbaseline, [1.0] * len(nonbaseline), -np.inf, float(global_max_nonbaseline))

    if use_config_max_uses:
        for _, (limit, indices) in sorted(_config_max_use_groups(rows).items()):
            add_constraint(indices, [1.0] * len(indices), -np.inf, float(limit))

    if use_cooldown:
        for one, two in _cooldown_conflicts(rows):
            add_constraint([one, two], [1.0, 1.0], -np.inf, 1.0)

    matrix = csr_matrix((values, (entries, columns)), shape=(len(lower), n))
    costs = np.asarray([row.cost for row in rows], dtype=float)
    tie_break = TIE_BREAK_EPS * costs + TIE_BREAK_EPS * np.asarray(
        [0.0 if int(row.is_baseline) == 1 else 1.0 for row in rows], dtype=float
    )
    objective = -np.asarray(utility, dtype=float) + tie_break
    options: Dict[str, Any] = {"mip_rel_gap": float(mip_gap)}
    if float(time_limit_sec) > 0:
        options["time_limit"] = float(time_limit_sec)

    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options=options,
    )
    if not bool(result.success):
        raise C5Error(f"SciPy/HiGHS MILP failed: {result.message}")

    selected = [index for index, value in enumerate(result.x) if float(value) > 0.5]
    feasibility = _allocation_feasibility(
        rows,
        selected,
        budget=budget,
        global_max_nonbaseline=global_max_nonbaseline,
        use_cooldown=use_cooldown,
        use_config_max_uses=use_config_max_uses,
    )
    if not feasibility["feasible"]:
        raise C5Error(f"MILP solver returned an infeasible allocation: {feasibility}")

    info = {
        "solver": "scipy.optimize.milp/HiGHS",
        "solver_message": str(result.message),
        "solver_status": int(result.status),
        "optimality_gap": float(getattr(result, "mip_gap", 0.0) or 0.0),
        "utility_objective": float(np.asarray(utility)[selected].sum()),
        **feasibility,
    }
    return selected, info


def _baseline_selection(rows: Sequence[RelativeRow]) -> List[int]:
    selected: List[int] = []
    for _, indices in _state_groups(rows).items():
        baseline = [index for index in indices if int(rows[index].is_baseline) == 1]
        if len(baseline) != 1:
            raise C5Error("Each state must have exactly one baseline action.")
        selected.append(baseline[0])
    return sorted(selected)




def _selection_map(rows: Sequence[RelativeRow], selected: Sequence[int]) -> Dict[str, str]:
    return {rows[index].state_id: rows[index].action_id for index in selected}


def _selection_difference(rows: Sequence[RelativeRow], one: Sequence[int], two: Sequence[int]) -> int:
    first = _selection_map(rows, one)
    second = _selection_map(rows, two)
    states = sorted(set(first) | set(second))
    return int(sum(first.get(state) != second.get(state) for state in states))


def _utility_array(rows: Sequence[RelativeRow], attr: str) -> np.ndarray:
    return np.asarray([float(getattr(row, attr)) for row in rows], dtype=float)


def _evaluate_selection(
    rows: Sequence[RelativeRow],
    selected: Sequence[int],
    method: str,
    budget_fraction: float,
    budget: Optional[float],
    global_max_nonbaseline: Optional[int],
    info: Mapping[str, Any],
    reference_total_gain: float,
    unconstrained_selected: Optional[Sequence[int]],
) -> Dict[str, Any]:
    n_states = len(_state_groups(rows))
    oracle_total_gain = float(sum(rows[index].oracle_gain for index in selected))
    constraint_changes = (
        0 if unconstrained_selected is None else _selection_difference(rows, selected, unconstrained_selected)
    )
    return {
        "seed": int(rows[0].seed),
        "method": method,
        "budget_fraction": float(budget_fraction),
        "budget": None if budget is None else float(budget),
        "n_states": int(n_states),
        "oracle_total_gain": oracle_total_gain,
        "oracle_gain_per_decision": float(oracle_total_gain / max(n_states, 1)),
        "reference_total_gain": float(reference_total_gain),
        "reference_minus_selected_total_gain": float(reference_total_gain - oracle_total_gain),
        "reference_gap_per_decision": float((reference_total_gain - oracle_total_gain) / max(n_states, 1)),
        "constraint_changed_decisions": int(constraint_changes),
        **dict(info),
    }



def _trajectory_allocation_analysis(
    rows: Sequence[RelativeRow],
    budget_fractions: Sequence[float],
    max_nonbaseline_fraction: float,
    time_limit_sec: float,
    mip_gap: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_seed: Dict[int, List[RelativeRow]] = defaultdict(list)
    for row in rows:
        by_seed[int(row.seed)].append(row)

    result_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    scale_rows: List[Dict[str, Any]] = []

    for seed, seed_rows in sorted(by_seed.items()):
        seed_rows = sorted(seed_rows, key=lambda row: (row.decision_day, row.state_id, row.action_id))
        n_states = len(_state_groups(seed_rows))
        if n_states == 0:
            continue

        nonbaseline_costs = [row.cost for row in seed_rows if int(row.is_baseline) == 0]
        max_action_cost = max(nonbaseline_costs) if nonbaseline_costs else 0.0
        global_max_nonbaseline = max(
            1,
            int(math.floor(float(n_states) * float(max_nonbaseline_fraction))),
        )

        utilities = {
            "calibrated_milp": _utility_array(seed_rows, "calibrated_gain"),
            "raw_score_milp": _utility_array(seed_rows, "raw_delta"),
            "state_normalised_milp": _utility_array(seed_rows, "state_normalised_score"),
            "oracle_reference_milp": _utility_array(seed_rows, "oracle_gain"),
        }

        # The manuscript reports changes relative to the unconstrained raw-score
        # allocation, so only that unconstrained solution is needed here.
        raw_unconstrained, _ = _solve_milp(
            seed_rows,
            utilities["raw_score_milp"],
            budget=None,
            global_max_nonbaseline=None,
            use_cooldown=False,
            use_config_max_uses=False,
            time_limit_sec=time_limit_sec,
            mip_gap=mip_gap,
        )

        for fraction in budget_fractions:
            budget = float(fraction) * float(n_states) * float(max_action_cost)
            selected_map: Dict[str, List[int]] = {}
            info_map: Dict[str, Dict[str, Any]] = {}

            for method in (
                "calibrated_milp",
                "raw_score_milp",
                "state_normalised_milp",
                "oracle_reference_milp",
            ):
                selected_map[method], info_map[method] = _solve_milp(
                    seed_rows,
                    utilities[method],
                    budget=budget,
                    global_max_nonbaseline=global_max_nonbaseline,
                    use_cooldown=True,
                    use_config_max_uses=True,
                    time_limit_sec=time_limit_sec,
                    mip_gap=mip_gap,
                )

            selected_map["baseline"] = _baseline_selection(seed_rows)
            info_map["baseline"] = {
                "solver": "fixed_baseline",
                "solver_message": "not_applicable",
                "solver_status": 0,
                "optimality_gap": float("nan"),
                "utility_objective": 0.0,
                **_allocation_feasibility(
                    seed_rows,
                    selected_map["baseline"],
                    budget=budget,
                    global_max_nonbaseline=global_max_nonbaseline,
                    use_cooldown=True,
                    use_config_max_uses=True,
                ),
            }

            reference_total_gain = float(
                sum(seed_rows[index].oracle_gain for index in selected_map["oracle_reference_milp"])
            )

            for method, selected in selected_map.items():
                unconstrained_selected = raw_unconstrained if method == "raw_score_milp" else None
                result = _evaluate_selection(
                    seed_rows,
                    selected,
                    method=method,
                    budget_fraction=float(fraction),
                    budget=budget,
                    global_max_nonbaseline=global_max_nonbaseline,
                    info=info_map[method],
                    reference_total_gain=reference_total_gain,
                    unconstrained_selected=unconstrained_selected,
                )
                result_rows.append(result)
                for index in selected:
                    row = seed_rows[index]
                    decision_rows.append({
                        "seed": seed,
                        "budget_fraction": float(fraction),
                        "method": method,
                        "state_id": row.state_id,
                        "decision_day": row.decision_day,
                        "action_id": row.action_id,
                        "action_name": row.action_name,
                        "action_display_name": row.action_display_name,
                        "is_baseline": row.is_baseline,
                        "cost": row.cost,
                        "raw_delta": row.raw_delta,
                        "calibrated_gain": row.calibrated_gain,
                        "state_normalised_score": row.state_normalised_score,
                        "oracle_gain": row.oracle_gain,
                    })

            # Score-scale sensitivity is reported relative to the validation-mapped
            # MILP. Only the original raw score and per-state normalised score are
            # retained as comparators.
            calibrated = selected_map["calibrated_milp"]
            calibrated_gain = (
                sum(seed_rows[index].oracle_gain for index in calibrated) / max(n_states, 1)
            )
            for comparator in ("raw_score_milp", "state_normalised_milp"):
                selected = selected_map[comparator]
                comparator_gain = (
                    sum(seed_rows[index].oracle_gain for index in selected) / max(n_states, 1)
                )
                changes = _selection_difference(seed_rows, calibrated, selected)
                scale_rows.append({
                    "seed": seed,
                    "budget_fraction": float(fraction),
                    "comparator": comparator,
                    "n_states": n_states,
                    "changed_decisions_vs_calibrated": int(changes),
                    "changed_fraction_vs_calibrated": float(changes / max(n_states, 1)),
                    "calibrated_oracle_gain_per_decision": float(calibrated_gain),
                    "comparator_oracle_gain_per_decision": float(comparator_gain),
                    "comparator_minus_calibrated_gain_per_decision": float(
                        comparator_gain - calibrated_gain
                    ),
                })

    return result_rows, decision_rows, scale_rows



def _bootstrap_mean(values: np.ndarray, repeats: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repeats), len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return float(low), float(high)


def _aggregate_allocations(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["budget_fraction"]), str(row["method"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (budget_fraction, method), group in sorted(groups.items()):
        gains = np.asarray([float(row["oracle_gain_per_decision"]) for row in group], dtype=float)
        gaps = np.asarray([float(row["reference_gap_per_decision"]) for row in group], dtype=float)
        gain_low, gain_high = _bootstrap_mean(gains, bootstrap_replicates, bootstrap_seed)
        gap_low, gap_high = _bootstrap_mean(gaps, bootstrap_replicates, bootstrap_seed + 13)
        finite_gaps = [float(row["optimality_gap"]) for row in group if math.isfinite(_safe_float(row["optimality_gap"]))]
        out.append({
            "budget_fraction": budget_fraction,
            "method": method,
            "n_independent_trajectories": len(group),
            "mean_oracle_gain_per_decision": float(gains.mean()),
            "oracle_gain_ci_low": gain_low,
            "oracle_gain_ci_high": gain_high,
            "mean_reference_gap_per_decision": float(gaps.mean()),
            "reference_gap_ci_low": gap_low,
            "reference_gap_ci_high": gap_high,
            "feasibility_rate": float(np.mean([1.0 if bool(row["feasible"]) else 0.0 for row in group])),
            "mean_total_cost": float(np.mean([float(row["total_cost"]) for row in group])),
            "mean_budget_slack": float(np.mean([float(row["budget_slack"]) for row in group if row["budget_slack"] is not None])),
            "mean_nonbaseline_uses": float(np.mean([float(row["nonbaseline_uses"]) for row in group])),
            "mean_max_nonbaseline_slack": float(np.mean([float(row["max_nonbaseline_slack"]) for row in group if row["max_nonbaseline_slack"] is not None])),
            "mean_constraint_changed_decisions": float(np.mean([float(row["constraint_changed_decisions"]) for row in group])),
            "total_cooldown_violations": int(sum(int(row["cooldown_violations"]) for row in group)),
            "total_config_max_use_violations": int(sum(int(row["config_max_use_violations"]) for row in group)),
            "max_solver_optimality_gap": max(finite_gaps) if finite_gaps else float("nan"),
        })
    return out


def _aggregate_scale(rows: Sequence[Mapping[str, Any]], bootstrap_replicates: int, bootstrap_seed: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["budget_fraction"]), str(row["comparator"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (budget, comparator), group in sorted(groups.items()):
        changed = np.asarray([float(row["changed_fraction_vs_calibrated"]) for row in group])
        delta_gain = np.asarray([float(row["comparator_minus_calibrated_gain_per_decision"]) for row in group])
        low, high = _bootstrap_mean(delta_gain, bootstrap_replicates, bootstrap_seed + 29)
        out.append({
            "budget_fraction": budget,
            "comparator": comparator,
            "n_independent_trajectories": len(group),
            "mean_changed_fraction_vs_calibrated": float(changed.mean()),
            "mean_comparator_minus_calibrated_gain_per_decision": float(delta_gain.mean()),
            "gain_difference_ci_low": low,
            "gain_difference_ci_high": high,
        })
    return out



def _plot_allocation_performance(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    methods = [
        "raw_score_milp",
        "calibrated_milp",
        "oracle_reference_milp",
        "baseline",
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for method in methods:
        rows = sorted(
            [row for row in summary if str(row["method"]) == method],
            key=lambda row: float(row["budget_fraction"]),
        )
        if not rows:
            continue
        x = np.asarray([float(row["budget_fraction"]) for row in rows])
        y = np.asarray([float(row["mean_oracle_gain_per_decision"]) for row in rows])
        low = np.asarray([_safe_float(row.get("oracle_gain_ci_low"), np.nan) for row in rows])
        high = np.asarray([_safe_float(row.get("oracle_gain_ci_high"), np.nan) for row in rows])
        yerr = None
        if np.isfinite(low).all() and np.isfinite(high).all():
            yerr = np.vstack([y - low, high - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.3,
            capsize=3,
            label=_method_label(method),
        )
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Simulator-derived gain per decision")
    ax.set_title("Constrained allocation performance")
    ax.legend(frameon=False, fontsize=8)
    _nature_axes(ax)
    _savefig600(path)




def _plot_reference_gap(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = sorted(
        [row for row in summary if str(row["method"]) == "raw_score_milp"],
        key=lambda row: float(row["budget_fraction"]),
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if rows:
        ax.plot(
            [float(row["budget_fraction"]) for row in rows],
            [float(row["mean_reference_gap_per_decision"]) for row in rows],
            marker="o",
            linewidth=1.3,
            label="raw-score MILP",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Reference minus raw-score MILP gain per decision")
    ax.set_title("Gap to finite-rollout constrained reference")
    ax.legend(frameon=False, fontsize=8)
    _nature_axes(ax)
    _savefig600(path)




def _plot_constraint_activity(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows = sorted(
        [row for row in summary if str(row["method"]) == "raw_score_milp"],
        key=lambda row: float(row["budget_fraction"]),
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    ax.bar(
        x - width / 2,
        [float(row["mean_nonbaseline_uses"]) for row in rows],
        width=width,
        label="Non-baseline uses",
    )
    ax.bar(
        x + width / 2,
        [float(row["mean_constraint_changed_decisions"]) for row in rows],
        width=width,
        label="Decisions changed vs unconstrained raw score",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(row['budget_fraction']):.2f}" for row in rows])
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Mean count per trajectory")
    ax.set_title("Constraint activity in raw-score MILP")
    ax.legend(frameon=False, fontsize=8)
    _nature_axes(ax)
    _savefig600(path)




def _plot_scale_sensitivity(scale_summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for comparator in sorted({str(row["comparator"]) for row in scale_summary}):
        rows = sorted(
            [row for row in scale_summary if str(row["comparator"]) == comparator],
            key=lambda row: float(row["budget_fraction"]),
        )
        ax.plot(
            [float(row["budget_fraction"]) for row in rows],
            [float(row["mean_changed_fraction_vs_calibrated"]) for row in rows],
            marker="o",
            linewidth=1.3,
            label=_method_label(comparator),
        )
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Fraction of decisions differing from validation-mapped MILP")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Cross-state score-scale sensitivity")
    ax.legend(frameon=False, fontsize=8)
    _nature_axes(ax)
    _savefig600(path)






def _plot_dashboard(
    calibration_rows: Sequence[RelativeRow],
    calibration_bins: Sequence[Mapping[str, Any]],
    allocation_summary: Sequence[Mapping[str, Any]],
    scale_summary: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    data = [row for row in calibration_rows if int(row.is_baseline) == 0]
    ax = axes[0, 0]
    ax.scatter(
        [row.calibrated_gain for row in data],
        [row.oracle_gain for row in data],
        s=10,
        alpha=0.18,
    )
    if calibration_bins:
        ax.plot(
            [float(row["predicted_mean"]) for row in calibration_bins],
            [float(row["observed_mean"]) for row in calibration_bins],
            "o-",
            linewidth=1.2,
        )
    finite = [
        value
        for row in data
        for value in (row.calibrated_gain, row.oracle_gain)
        if math.isfinite(value)
    ]
    if finite:
        lo, hi = min(finite), max(finite)
        if math.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Validation-mapped predicted gain")
    ax.set_ylabel("Observed simulator gain")
    ax.set_title("A. Cross-state calibration")
    _nature_axes(ax)

    ax = axes[0, 1]
    for method in ("raw_score_milp", "oracle_reference_milp", "baseline"):
        rows = sorted(
            [row for row in allocation_summary if str(row["method"]) == method],
            key=lambda row: float(row["budget_fraction"]),
        )
        if rows:
            ax.plot(
                [float(row["budget_fraction"]) for row in rows],
                [float(row["mean_oracle_gain_per_decision"]) for row in rows],
                marker="o",
                label=_method_label(method),
            )
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Gain per decision")
    ax.set_title("B. Raw-score constrained allocation")
    ax.legend(frameon=False, fontsize=7)
    _nature_axes(ax)

    ax = axes[1, 0]
    rows = sorted(
        [row for row in allocation_summary if str(row["method"]) == "raw_score_milp"],
        key=lambda row: float(row["budget_fraction"]),
    )
    x = np.arange(len(rows))
    ax.bar(x, [float(row["mean_constraint_changed_decisions"]) for row in rows])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(row['budget_fraction']):.2f}" for row in rows])
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Changed decisions")
    ax.set_title("C. Raw-score MILP constraint activity")
    _nature_axes(ax)

    ax = axes[1, 1]
    for comparator in sorted({str(row["comparator"]) for row in scale_summary}):
        rows = sorted(
            [row for row in scale_summary if str(row["comparator"]) == comparator],
            key=lambda row: float(row["budget_fraction"]),
        )
        ax.plot(
            [float(row["budget_fraction"]) for row in rows],
            [float(row["mean_changed_fraction_vs_calibrated"]) for row in rows],
            marker="o",
            label=_method_label(comparator),
        )
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Changed fraction")
    ax.set_title("D. Score-scale sensitivity")
    ax.legend(frameon=False, fontsize=7)
    _nature_axes(ax)

    fig.tight_layout()
    _savefig600(path)



def _fmt(value: Any, digits: int = 3) -> str:
    number = _safe_float(value, float("nan"))
    if not math.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _ci_text(row: Mapping[str, Any], mean_key: str, low_key: str, high_key: str) -> str:
    mean = _fmt(row.get(mean_key))
    low = _fmt(row.get(low_key))
    high = _fmt(row.get(high_key))
    return f"{mean} [{low}, {high}]"



def _write_calibration_table(path: Path, diagnostics: Sequence[Mapping[str, Any]]) -> None:
    raw = next(
        row for row in diagnostics
        if str(row["variant"]) == "raw_score_delta" and str(row["subset"]) == "nonbaseline"
    )
    mapped = next(
        row for row in diagnostics
        if str(row["variant"]) == "isotonic_calibrated" and str(row["subset"]) == "nonbaseline"
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Cross-state score diagnostics for the C5 allocation evaluation. Raw-score ordering accuracy is calculated from the original baseline-relative policy scores across non-tied pairs of non-baseline actions belonging to different test states. The remaining quantities evaluate the validation-fitted isotonic score-to-gain mapping on held-out non-baseline state--action rows. Bias is predicted minus simulator-derived gain.}",
        r"\label{tab:c5_calibration}",
        r"\begin{tabular}{rrrrrrr}",
        r"\hline",
        r"Raw order acc. & Mapped RMSE & Mapped MAE & Mapped bias & Cal. slope & Spearman $\rho$ & Pairwise pairs \\",
        r"\hline",
        f"{_fmt(raw['cross_state_pairwise_accuracy'])} & {_fmt(mapped['rmse'])} & {_fmt(mapped['mae'])} & "
        f"{_fmt(mapped['bias_pred_minus_observed'])} & {_fmt(mapped['calibration_slope_observed_on_predicted'])} & "
        f"{_fmt(mapped['spearman_rho'])} & {int(raw['cross_state_pairwise_pairs'])} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _find_summary_row(summary: Sequence[Mapping[str, Any]], budget: float, method: str) -> Mapping[str, Any]:
    for row in summary:
        if math.isclose(float(row["budget_fraction"]), float(budget), abs_tol=1e-12) and str(row["method"]) == method:
            return row
    raise C5Error(f"Missing allocation summary row for budget={budget}, method={method}")



def _write_primary_allocation_table(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    budgets = sorted({float(row["budget_fraction"]) for row in summary})
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Quantitative evaluation of the original raw-score MILP under active resource and cooldown constraints. Gain is the simulator-derived baseline-relative reduction in resistant events per decision; larger values are preferable. Confidence intervals are obtained by bootstrapping the independent test trajectories. The finite-rollout reference maximises realised simulator-derived gains under the same constraints and is not an estimate of the true expected-benefit optimum. Changed decisions is the mean number of selected actions that differ from the corresponding unconstrained raw-score allocation.}",
        r"\label{tab:c5_allocation}",
        r"\begin{tabular}{rrrrr}",
        r"\hline",
        r"Budget & Raw-score MILP gain [95\% CI] & Finite reference & Reference gap & Changed decisions \\",
        r"\hline",
    ]
    for budget in budgets:
        raw = _find_summary_row(summary, budget, "raw_score_milp")
        reference = _find_summary_row(summary, budget, "oracle_reference_milp")
        lines.append(
            f"{budget:.2f} & "
            f"{_ci_text(raw, 'mean_oracle_gain_per_decision', 'oracle_gain_ci_low', 'oracle_gain_ci_high')} & "
            f"{_fmt(reference['mean_oracle_gain_per_decision'])} & "
            f"{_fmt(raw['mean_reference_gap_per_decision'])} & "
            f"{_fmt(raw['mean_constraint_changed_decisions'], 2)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")






def _write_constraint_table(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    rows = sorted(
        [row for row in summary if str(row["method"]) == "raw_score_milp"],
        key=lambda row: float(row["budget_fraction"]),
    )
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Constraint and solver diagnostics for the original raw-score C5 MILP under the active stress-test restrictions.}",
        r"\label{tab:c5_constraints}",
        r"\begin{tabular}{rrrrrrr}",
        r"\hline",
        r"Budget & Feasible & Uses & Budget slack & Changed decisions & Cooldown violations & Max MIP gap \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{float(row['budget_fraction']):.2f} & {_fmt(row['feasibility_rate'])} & "
            f"{_fmt(row['mean_nonbaseline_uses'], 2)} & {_fmt(row['mean_budget_slack'], 2)} & "
            f"{_fmt(row['mean_constraint_changed_decisions'], 2)} & "
            f"{int(row['total_cooldown_violations'])} & {_fmt(row['max_solver_optimality_gap'], 6)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _write_scale_table(path: Path, scale_summary: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Sensitivity of constrained allocation to cross-state score scale. Each comparator is evaluated against the validation-calibrated MILP while preserving the same decision states and operational constraints.}",
        r"\label{tab:c5_scale_sensitivity}",
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Budget & Comparator & Changed decision fraction & Gain difference vs calibrated \\",
        r"\hline",
    ]
    for row in sorted(scale_summary, key=lambda item: (float(item["budget_fraction"]), str(item["comparator"]))):
        lines.append(
            f"{float(row['budget_fraction']):.2f} & {_latex_escape(str(row['comparator']).replace('_', ' '))} & "
            f"{_fmt(row['mean_changed_fraction_vs_calibrated'])} & {_fmt(row['mean_comparator_minus_calibrated_gain_per_decision'])} \\\\" 
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _write_figure_captions(path: Path) -> None:
    lines = [
        r"\paragraph{Figure C5.1 --- Cross-state benefit calibration.} Each point is a non-baseline state--action pair from the untouched test split. The horizontal coordinate is the validation-fitted isotonic transformation of the original baseline-relative policy score into simulator-gain units; the vertical coordinate is the simulator-derived baseline-relative gain. Binned means summarise calibration across the score range.",
        "",
        r"\paragraph{Figure C5.2 --- Constrained allocation performance.} Mean simulator-derived gain per decision across independent test trajectories for the original raw-score MILP, validation-mapped sensitivity MILP, finite-rollout simulator-reference MILP, and baseline allocation across resource budgets. Error bars are trajectory-bootstrap 95\% confidence intervals where defined.",
        "",
        r"\paragraph{Figure C5.3 --- Gap to constrained reference.} Difference between the finite-rollout reference allocation and the original raw-score MILP in simulator-derived gain per decision. Zero indicates equality with the finite-rollout reference under the same constraints.",
        "",
        r"\paragraph{Figure C5.4 --- Constraint activity.} Mean number of non-baseline interventions selected by the raw-score MILP and mean number of decisions changed relative to its corresponding unconstrained raw-score allocation at each budget level.",
        "",
        r"\paragraph{Figure C5.5 --- Cross-state score-scale sensitivity.} Fraction of decision states whose selected action changes relative to the validation-mapped MILP when optimisation instead uses the original raw score or a per-state normalised score that preserves within-state action ordering.",
        "",
        r"\paragraph{Figure C5.6 --- C5 dashboard.} Summary of validation-only score mapping, original raw-score constrained allocation, raw-score constraint activity, and score-scale sensitivity.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _c4_reproduction_check(
    test_rows: Sequence[RelativeRow],
    c4_milp_dir: Path,
    time_limit_sec: float,
    mip_gap: float,
) -> Dict[str, Any]:
    plan_path = c4_milp_dir / "milp_policy_plan.csv"
    if not plan_path.is_file():
        return {"status": "not_available", "reason": f"Missing {plan_path}"}

    c4_rows = _read_csv(plan_path)
    c4_map = {
        str(row.get("state_id", "")).strip(): str(
            row.get("chosen_action_id")
            or row.get("action_id")
            or row.get("selected_action_id")
            or ""
        ).strip()
        for row in c4_rows
        if str(row.get("state_id", "")).strip()
    }
    if not c4_map:
        return {"status": "not_available", "reason": "Could not parse state_id/action_id from C4 plan."}

    by_seed: Dict[int, List[RelativeRow]] = defaultdict(list)
    for row in test_rows:
        by_seed[row.seed].append(row)

    c5_map: Dict[str, str] = {}
    c5_objective = 0.0
    for _, rows in sorted(by_seed.items()):
        rows = sorted(rows, key=lambda row: (row.decision_day, row.state_id, row.action_id))
        utility = _utility_array(rows, "raw_delta")
        selected, info = _solve_milp(
            rows,
            utility,
            budget=None,
            global_max_nonbaseline=None,
            use_cooldown=True,
            use_config_max_uses=True,
            time_limit_sec=time_limit_sec,
            mip_gap=mip_gap,
        )
        c5_objective += float(info["utility_objective"])
        c5_map.update(_selection_map(rows, selected))

    shared_states = sorted(set(c4_map) & set(c5_map))
    if not shared_states:
        return {"status": "not_comparable", "reason": "C4 and C5 plans share no parsed state IDs."}
    matches = sum(c4_map[state] == c5_map[state] for state in shared_states)

    raw_lookup = {(row.state_id, row.action_id): row.raw_delta for row in test_rows}
    c4_objective = float(sum(raw_lookup.get((state, c4_map[state]), 0.0) for state in shared_states))
    c5_shared_objective = float(sum(raw_lookup.get((state, c5_map[state]), 0.0) for state in shared_states))
    return {
        "status": "complete",
        "n_shared_states": len(shared_states),
        "selection_match_count": int(matches),
        "selection_match_rate": float(matches / len(shared_states)),
        "c4_raw_score_objective_on_shared_states": c4_objective,
        "c5_raw_score_objective_on_shared_states": c5_shared_objective,
        "objective_difference_c5_minus_c4": float(c5_shared_objective - c4_objective),
        "c5_total_raw_score_objective": float(c5_objective),
        "interpretation": "Different selections can still be objective-equivalent when action-score ties exist.",
    }


def _write_relative_rows(path: Path, rows: Sequence[RelativeRow]) -> None:
    _write_csv(path, [row.__dict__ for row in rows])



def _primary_numbers(
    diagnostics: Sequence[Mapping[str, Any]],
    allocation_summary: Sequence[Mapping[str, Any]],
    scale_summary: Sequence[Mapping[str, Any]],
    c4_check: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_diag = next(
        (row for row in diagnostics if str(row["variant"]) == "raw_score_delta" and str(row["subset"]) == "nonbaseline"),
        {},
    )
    mapped_diag = next(
        (row for row in diagnostics if str(row["variant"]) == "isotonic_calibrated" and str(row["subset"]) == "nonbaseline"),
        {},
    )
    budgets = sorted({float(row["budget_fraction"]) for row in allocation_summary})
    allocation: Dict[str, Any] = {}
    for budget in budgets:
        raw = _find_summary_row(allocation_summary, budget, "raw_score_milp")
        reference = _find_summary_row(allocation_summary, budget, "oracle_reference_milp")
        baseline = _find_summary_row(allocation_summary, budget, "baseline")
        calibrated = _find_summary_row(allocation_summary, budget, "calibrated_milp")
        normalised = _find_summary_row(allocation_summary, budget, "state_normalised_milp")
        allocation[f"budget_{budget:.2f}"] = {
            "raw_score_gain_per_decision": raw["mean_oracle_gain_per_decision"],
            "raw_score_gain_ci": [raw["oracle_gain_ci_low"], raw["oracle_gain_ci_high"]],
            "reference_gain_per_decision": reference["mean_oracle_gain_per_decision"],
            "raw_score_reference_gap_per_decision": raw["mean_reference_gap_per_decision"],
            "baseline_gain_per_decision": baseline["mean_oracle_gain_per_decision"],
            "validation_mapped_gain_per_decision": calibrated["mean_oracle_gain_per_decision"],
            "state_normalised_gain_per_decision": normalised["mean_oracle_gain_per_decision"],
            "raw_score_feasibility_rate": raw["feasibility_rate"],
            "raw_score_mean_constraint_changed_decisions": raw["mean_constraint_changed_decisions"],
            "raw_score_cooldown_violations": raw["total_cooldown_violations"],
            "raw_score_max_solver_optimality_gap": raw["max_solver_optimality_gap"],
        }
    return {
        "raw_score_cross_state_ordering": dict(raw_diag),
        "validation_mapped_calibration": dict(mapped_diag),
        "allocation": allocation,
        "scale_sensitivity": list(scale_summary),
        "c4_reproduction_check": dict(c4_check),
    }




def _run_single_track(args: argparse.Namespace) -> int:
    data_folder = Path(args.data_folder).resolve()
    trained_dir = Path(args.trained_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()
    c4_milp_dir = Path(args.c4_milp_dir).resolve()
    milp_config_path = Path(args.milp_config_json).resolve()
    candidate_json = Path(args.candidate_interventions_json).resolve()
    evaluator_script = Path(args.evaluator_script).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in (data_folder, trained_dir, eval_dir):
        if not required.exists():
            raise C5Error(f"Required C1--C4 path does not exist: {required}")
    for required_file in (milp_config_path, candidate_json, evaluator_script):
        if not required_file.is_file():
            raise C5Error(f"Required file does not exist: {required_file}")

    test_summary_path = eval_dir / "policy_selection_summary.json"
    if not test_summary_path.is_file():
        raise C5Error(f"C3 summary not found: {test_summary_path}")
    test_summary = _read_json(test_summary_path)
    oracle_metric = str(test_summary.get("oracle_metric", "")).strip()
    if not oracle_metric:
        raise C5Error("C3 policy_selection_summary.json does not define oracle_metric.")
    oracle_direction = _oracle_direction(test_summary)

    test_score_path = eval_dir / "policy_action_scores_requested_split.csv"
    if not test_score_path.is_file():
        test_score_path = eval_dir / "policy_action_scores.csv"
    test_scores = _load_score_rows(test_score_path, required_split=args.evaluation_split)

    validation_dir = out_dir / "validation_scoring"
    validation_score_path = _ensure_validation_scores(
        evaluator_script=evaluator_script,
        data_folder=data_folder,
        trained_dir=trained_dir,
        candidate_interventions_json=candidate_json,
        out_dir=validation_dir,
        split=args.calibration_split,
        oracle_metric=oracle_metric,
        selection_direction="auto",
        force=bool(args.force_validation_rescore),
    )
    validation_scores = _load_score_rows(
        validation_score_path,
        required_split=args.calibration_split,
    )

    validation_relative = _relative_rows(validation_scores, oracle_direction=oracle_direction)
    test_relative = _relative_rows(test_scores, oracle_direction=oracle_direction)

    isotonic, zero_anchor = _fit_isotonic_calibrator(validation_relative)
    _apply_isotonic_calibrator(validation_relative, isotonic, zero_anchor)
    _apply_isotonic_calibrator(test_relative, isotonic, zero_anchor)

    milp_config = _read_json(milp_config_path)
    _apply_rules(validation_relative, milp_config)
    _apply_rules(test_relative, milp_config)

    diagnostics: List[Dict[str, Any]] = []
    for subset in ("all", "nonbaseline"):
        diagnostics.append(_diagnostic_row(test_relative, "raw_delta", "raw_score_delta", subset))
        diagnostics.append(
            _diagnostic_row(test_relative, "calibrated_gain", "isotonic_calibrated", subset)
        )

    bins = _calibration_bins(test_relative, int(args.calibration_bins))
    budget_fractions = _parse_float_list(args.budget_fractions)
    allocation_rows, decision_rows, scale_rows = _trajectory_allocation_analysis(
        test_relative,
        budget_fractions=budget_fractions,
        max_nonbaseline_fraction=float(args.max_nonbaseline_fraction),
        time_limit_sec=float(args.time_limit_sec),
        mip_gap=float(args.mip_gap),
    )
    allocation_summary = _aggregate_allocations(
        allocation_rows,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    scale_summary = _aggregate_scale(
        scale_rows,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    c4_check = _c4_reproduction_check(
        test_relative,
        c4_milp_dir=c4_milp_dir,
        time_limit_sec=float(args.time_limit_sec),
        mip_gap=float(args.mip_gap),
    )

    _write_relative_rows(out_dir / "validation_relative_scores.csv", validation_relative)
    _write_relative_rows(out_dir / "test_relative_scores.csv", test_relative)
    _write_csv(out_dir / "calibration_diagnostics.csv", diagnostics)
    _write_csv(out_dir / "calibration_bins.csv", bins)
    _write_csv(out_dir / "allocation_by_trajectory.csv", allocation_rows)
    _write_csv(out_dir / "allocation_decisions.csv", decision_rows)
    _write_csv(out_dir / "allocation_summary.csv", allocation_summary)
    _write_csv(out_dir / "scale_sensitivity_by_trajectory.csv", scale_rows)
    _write_csv(out_dir / "scale_sensitivity_summary.csv", scale_summary)
    _write_json(out_dir / "c4_reproduction_check.json", c4_check)

    figures = out_dir / "figures"
    _plot_calibration(test_relative, bins, figures / "c5_calibration.png")
    _plot_allocation_performance(allocation_summary, figures / "c5_allocation_performance.png")
    _plot_reference_gap(allocation_summary, figures / "c5_reference_gap.png")
    _plot_constraint_activity(allocation_summary, figures / "c5_constraint_activity.png")
    _plot_scale_sensitivity(scale_summary, figures / "c5_scale_sensitivity.png")
    _plot_dashboard(test_relative, bins, allocation_summary, scale_summary, figures / "c5_dashboard.png")

    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_calibration_table(tables / "c5_calibration_table.tex", diagnostics)
    _write_primary_allocation_table(tables / "c5_allocation_table.tex", allocation_summary)
    _write_constraint_table(tables / "c5_constraint_table.tex", allocation_summary)
    _write_scale_table(tables / "c5_scale_sensitivity_table.tex", scale_summary)
    _write_figure_captions(tables / "c5_figure_captions.tex")

    numbers = _primary_numbers(diagnostics, allocation_summary, scale_summary, c4_check)
    _write_json(out_dir / "manuscript_numbers.json", numbers)

    raw_diagnostic = next(
        row for row in diagnostics
        if str(row["variant"]) == "raw_score_delta" and str(row["subset"]) == "nonbaseline"
    )
    calibration_summary = next(
        row for row in diagnostics
        if str(row["variant"]) == "isotonic_calibrated" and str(row["subset"]) == "nonbaseline"
    )
    c5_summary = {
        "status": "complete",
        "analysis_scope": "retrospective fixed-state constrained allocation over existing C1-C4 outputs",
        "original_causal_experiment_modified": False,
        "calibration_split": str(args.calibration_split),
        "evaluation_split": str(args.evaluation_split),
        "oracle_metric": oracle_metric,
        "oracle_direction": oracle_direction,
        "n_validation_rows": len(validation_relative),
        "n_test_rows": len(test_relative),
        "n_validation_states": len(_state_groups(validation_relative)),
        "n_test_states": len(_state_groups(test_relative)),
        "n_test_trajectories": len({row.seed for row in test_relative}),
        "isotonic_zero_anchor_before_reanchoring": zero_anchor,
        "raw_score_cross_state_diagnostics": raw_diagnostic,
        "primary_calibration_test_diagnostics": calibration_summary,
        "budget_fractions": budget_fractions,
        "max_nonbaseline_fraction": float(args.max_nonbaseline_fraction),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "reported_allocation_methods": [
            "raw_score_milp",
            "baseline",
            "oracle_reference_milp",
            "calibrated_milp",
            "state_normalised_milp",
        ],
        "finite_rollout_reference_caveat": (
            "Reference MILP uses existing simulator-derived test outcomes and is not the true "
            "expected-benefit optimum."
        ),
        "prospective_control_caveat": (
            "C5 is a retrospective fixed-state allocation benchmark and does not establish "
            "prospective adaptive control."
        ),
        "c4_reproduction_check": c4_check,
        "files": {
            "calibration_diagnostics": str(out_dir / "calibration_diagnostics.csv"),
            "allocation_summary": str(out_dir / "allocation_summary.csv"),
            "scale_sensitivity_summary": str(out_dir / "scale_sensitivity_summary.csv"),
            "manuscript_numbers": str(out_dir / "manuscript_numbers.json"),
            "figures_dir": str(figures),
            "tables_dir": str(tables),
        },
    }
    _write_json(out_dir / "c5_summary.json", c5_summary)

    manifest = {
        "main_text_candidates": [
            str(tables / "c5_allocation_table.tex"),
        ],
        "supplement_candidates": [
            str(figures / "c5_dashboard.png"),
            str(figures / "c5_calibration.png"),
            str(figures / "c5_reference_gap.png"),
            str(figures / "c5_constraint_activity.png"),
            str(figures / "c5_scale_sensitivity.png"),
            str(tables / "c5_calibration_table.tex"),
            str(tables / "c5_constraint_table.tex"),
            str(tables / "c5_scale_sensitivity_table.tex"),
            str(tables / "c5_figure_captions.tex"),
        ],
        "machine_readable": [
            str(out_dir / "calibration_diagnostics.csv"),
            str(out_dir / "calibration_bins.csv"),
            str(out_dir / "allocation_by_trajectory.csv"),
            str(out_dir / "allocation_decisions.csv"),
            str(out_dir / "allocation_summary.csv"),
            str(out_dir / "scale_sensitivity_summary.csv"),
            str(out_dir / "manuscript_numbers.json"),
            str(out_dir / "c5_summary.json"),
        ],
    }
    _write_json(out_dir / "manuscript_manifest.json", manifest)

    print(f"C5_MILP_EVALUATION_OK out_dir={out_dir}", flush=True)
    return 0



def _load_track_csv(track_c5_dir: Path, filename: str, state_mode: str) -> List[Dict[str, Any]]:
    path = track_c5_dir / filename
    rows = _read_csv(path)
    return [{**row, "state_mode": state_mode} for row in rows]



def _write_combined_allocation_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Quantitative evaluation of the original raw-score MILP under active resource and cooldown constraints. Gain is the simulator-derived baseline-relative reduction in resistant events per decision; larger values are preferable. Confidence intervals are obtained by bootstrapping the 10 independent test trajectories. The finite-rollout reference maximises realised simulator-derived gains under the same constraints and is not an estimate of the true expected-benefit optimum. Changed decisions is the mean number of selected actions, among seven retained states per trajectory, that differ from the corresponding unconstrained raw-score allocation.}",
        r"\label{tab:c5_combined_allocation}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"State representation & Budget & Raw-score MILP gain [95\% CI] & Finite reference & Reference gap & Changed decisions \\",
        r"\hline",
    ]
    modes = sorted({str(row["state_mode"]) for row in rows})
    budgets = sorted({float(row["budget_fraction"]) for row in rows})
    for state_mode in modes:
        mode = "Ground truth" if state_mode == "ground_truth" else "Partial observation"
        mode_rows = [row for row in rows if str(row["state_mode"]) == state_mode]
        for budget in budgets:
            raw = _find_summary_row(mode_rows, budget, "raw_score_milp")
            reference = _find_summary_row(mode_rows, budget, "oracle_reference_milp")
            lines.append(
                f"{mode} & {budget:.2f} & "
                f"{_ci_text(raw, 'mean_oracle_gain_per_decision', 'oracle_gain_ci_low', 'oracle_gain_ci_high')} & "
                f"{_fmt(reference['mean_oracle_gain_per_decision'])} & "
                f"{_fmt(raw['mean_reference_gap_per_decision'])} & "
                f"{_fmt(raw['mean_constraint_changed_decisions'], 2)} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")






def _write_combined_calibration_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Cross-state score diagnostics for the C5 allocation evaluation. Raw-score ordering accuracy is calculated from the original baseline-relative policy scores across non-tied pairs of non-baseline actions belonging to different test states. The remaining quantities evaluate the validation-fitted isotonic score-to-gain mapping on held-out non-baseline state--action rows. Bias is predicted minus simulator-derived gain.}",
        r"\label{tab:c5_combined_calibration}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"State representation & Raw order acc. & Mapped RMSE & Mapped MAE & Mapped bias & Cal. slope & Spearman $\rho$ \\",
        r"\hline",
    ]
    for state_mode in sorted({str(row["state_mode"]) for row in rows}):
        mode = "Ground truth" if state_mode == "ground_truth" else "Partial observation"
        mode_rows = [row for row in rows if str(row["state_mode"]) == state_mode]
        raw = next(
            row for row in mode_rows
            if str(row["variant"]) == "raw_score_delta" and str(row["subset"]) == "nonbaseline"
        )
        mapped = next(
            row for row in mode_rows
            if str(row["variant"]) == "isotonic_calibrated" and str(row["subset"]) == "nonbaseline"
        )
        lines.append(
            f"{mode} & {_fmt(raw['cross_state_pairwise_accuracy'])} & {_fmt(mapped['rmse'])} & "
            f"{_fmt(mapped['mae'])} & {_fmt(mapped['bias_pred_minus_observed'])} & "
            f"{_fmt(mapped['calibration_slope_observed_on_predicted'])} & {_fmt(mapped['spearman_rho'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")




def _plot_combined_primary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    for state_mode in sorted({str(row["state_mode"]) for row in rows}):
        for method in ("raw_score_milp", "oracle_reference_milp"):
            subset = sorted(
                [
                    row for row in rows
                    if str(row["state_mode"]) == state_mode and str(row["method"]) == method
                ],
                key=lambda row: float(row["budget_fraction"]),
            )
            if not subset:
                continue
            label_mode = "GT" if state_mode == "ground_truth" else "PO"
            label_method = "raw-score MILP" if method == "raw_score_milp" else "finite reference"
            ax.plot(
                [float(row["budget_fraction"]) for row in subset],
                [float(row["mean_oracle_gain_per_decision"]) for row in subset],
                marker="o",
                linewidth=1.4,
                label=f"{label_mode} {label_method}",
            )
    ax.set_xlabel("Budget fraction")
    ax.set_ylabel("Simulator-derived gain per decision")
    ax.set_title("C5 constrained allocation across observation settings")
    ax.legend(frameon=False, fontsize=8)
    _nature_axes(ax)
    _savefig600(path)




def _collate_parent(args: argparse.Namespace) -> int:
    parent = Path(args.collate_parent).resolve()
    out_dir = (
        Path(args.out_dir).resolve()
        if str(args.out_dir).strip()
        else parent / "C5_MANUSCRIPT_BUNDLE"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    track_specs = [
        ("ground_truth", parent / "TRACK_ground_truth" / "work" / args.c5_dir_name),
        ("partial_observation", parent / "TRACK_partial_observation" / "work" / args.c5_dir_name),
    ]
    for _, directory in track_specs:
        if not (directory / "c5_summary.json").is_file():
            raise C5Error(f"Cannot collate C5: missing {directory / 'c5_summary.json'}")

    allocation_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    scale_rows: List[Dict[str, Any]] = []
    combined_numbers: Dict[str, Any] = {}
    for state_mode, directory in track_specs:
        allocation_rows.extend(_load_track_csv(directory, "allocation_summary.csv", state_mode))
        calibration_rows.extend(_load_track_csv(directory, "calibration_diagnostics.csv", state_mode))
        scale_rows.extend(_load_track_csv(directory, "scale_sensitivity_summary.csv", state_mode))
        combined_numbers[state_mode] = _read_json(directory / "manuscript_numbers.json")
        track_dst = out_dir / state_mode
        track_dst.mkdir(parents=True, exist_ok=True)
        for subdir in ("figures", "tables"):
            src = directory / subdir
            if src.is_dir():
                shutil.copytree(src, track_dst / subdir, dirs_exist_ok=True)

    _write_csv(out_dir / "allocation_summary_combined.csv", allocation_rows)
    _write_csv(out_dir / "calibration_diagnostics_combined.csv", calibration_rows)
    _write_csv(out_dir / "scale_sensitivity_combined.csv", scale_rows)
    _write_json(out_dir / "manuscript_numbers_combined.json", combined_numbers)

    _write_combined_allocation_table(
        out_dir / "c5_combined_allocation_table.tex",
        allocation_rows,
    )
    _write_combined_calibration_table(
        out_dir / "c5_combined_calibration_table.tex",
        calibration_rows,
    )
    _plot_combined_primary(
        allocation_rows,
        out_dir / "c5_combined_allocation_performance.png",
    )

    material_lines = [
        "% Auto-generated C5 manuscript bundle",
        r"\input{c5_combined_calibration_table.tex}",
        "",
        r"\input{c5_combined_allocation_table.tex}",
        "",
        r"\begin{figure}[!t]",
        r"\centering",
        r"\includegraphics[width=0.82\textwidth]{c5_combined_allocation_performance.png}",
        r"\caption{Quantitative C5 constrained-allocation performance across the ground-truth and partial-observation tracks. The original raw-score MILP is shown against the finite-rollout simulator-reference allocation under the same active resource and cooldown constraints.}",
        r"\label{fig:c5_combined_allocation}",
        r"\end{figure}",
    ]
    (out_dir / "manuscript_materials.tex").write_text(
        "\n".join(material_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(
        out_dir / "bundle_manifest.json",
        {
            "tracks": {state_mode: str(directory) for state_mode, directory in track_specs},
            "combined_allocation_table": str(out_dir / "c5_combined_allocation_table.tex"),
            "combined_calibration_table": str(out_dir / "c5_combined_calibration_table.tex"),
            "combined_figure": str(out_dir / "c5_combined_allocation_performance.png"),
            "manuscript_materials": str(out_dir / "manuscript_materials.tex"),
        },
    )
    print(f"C5_COLLATE_OK out_dir={out_dir}", flush=True)
    return 0




def _self_test() -> int:
    rows: List[RelativeRow] = []
    for day, gains, raw in [
        (7, [0.0, 4.0, 3.0], [0.0, 0.4, 0.3]),
        (14, [0.0, 8.0, 2.0], [0.0, 0.8, 0.2]),
        (21, [0.0, 3.0, 4.0], [0.0, 0.3, 0.4]),
    ]:
        for action_index in range(3):
            rows.append(RelativeRow(
                state_id=f"S{day}",
                split="test",
                seed=1,
                decision_day=day,
                action_id=f"A{action_index}",
                action_name="screen",
                action_display_name=f"A{action_index}",
                is_baseline=1 if action_index == 0 else 0,
                pred_score=raw[action_index],
                baseline_pred_score=0.0,
                raw_delta=raw[action_index],
                oracle_value=gains[action_index],
                baseline_oracle_value=0.0,
                oracle_gain=gains[action_index],
                calibrated_gain=gains[action_index],
                state_normalised_score=raw[action_index],
                cost=0.0 if action_index == 0 else 1.0,
                cooldown_days=8 if action_index else 0,
                cooldown_group="screen" if action_index else "baseline",
                max_uses_group="screen" if action_index else "baseline",
            ))

    raw_utility = _utility_array(rows, "raw_delta")
    selected, info = _solve_milp(
        rows,
        raw_utility,
        budget=1.0,
        global_max_nonbaseline=1,
        use_cooldown=True,
        use_config_max_uses=True,
        time_limit_sec=30,
        mip_gap=0.0,
    )
    chosen = [rows[index] for index in selected if int(rows[index].is_baseline) == 0]
    assert info["feasible"]
    assert len(chosen) == 1 and chosen[0].decision_day == 14 and chosen[0].action_id == "A1"

    baseline = _baseline_selection(rows)
    assert _allocation_feasibility(
        rows,
        baseline,
        budget=1.0,
        global_max_nonbaseline=1,
        use_cooldown=True,
        use_config_max_uses=True,
    )["feasible"]

    invalid = [1, 4, 7]
    assert not _allocation_feasibility(
        rows,
        invalid,
        budget=1.0,
        global_max_nonbaseline=1,
        use_cooldown=True,
        use_config_max_uses=True,
    )["feasible"]

    print("C5_SELF_TEST_OK", flush=True)
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C5 quantitative cross-state calibration and MILP evaluation.")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--trained_dir", type=str, default="")
    parser.add_argument("--eval_dir", type=str, default="")
    parser.add_argument("--c4_milp_dir", type=str, default="")
    parser.add_argument("--milp_config_json", type=str, default="")
    parser.add_argument("--candidate_interventions_json", type=str, default="")
    parser.add_argument("--evaluator_script", type=str, default="evaluate_policy_selector.py")
    parser.add_argument("--out_dir", type=str, default="policy_milp_quantitative_eval")
    parser.add_argument("--calibration_split", type=str, default="validation")
    parser.add_argument("--evaluation_split", type=str, default="test")
    parser.add_argument("--budget_fractions", type=str, default="0.25,0.50,1.00")
    parser.add_argument("--max_nonbaseline_fraction", type=float, default=0.50)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=260928)
    parser.add_argument("--calibration_bins", type=int, default=8)
    parser.add_argument("--time_limit_sec", type=float, default=300.0)
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--force_validation_rescore", type=_str2bool, default=False)
    parser.add_argument("--collate_parent", type=str, default="")
    parser.add_argument("--c5_dir_name", type=str, default="policy_milp_quantitative_eval")
    parser.add_argument("--self_test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return _self_test()
    if str(args.collate_parent).strip():
        return _collate_parent(args)
    required = {
        "--data_folder": args.data_folder,
        "--trained_dir": args.trained_dir,
        "--eval_dir": args.eval_dir,
        "--c4_milp_dir": args.c4_milp_dir,
        "--milp_config_json": args.milp_config_json,
        "--candidate_interventions_json": args.candidate_interventions_json,
    }
    missing = [flag for flag, value in required.items() if not str(value).strip()]
    if missing:
        raise C5Error("Missing required arguments for single-track C5 run: " + ", ".join(missing))
    if not 0 < float(args.max_nonbaseline_fraction) <= 1:
        raise C5Error("--max_nonbaseline_fraction must be in (0, 1].")
    if int(args.bootstrap_replicates) < 100:
        raise C5Error("--bootstrap_replicates must be at least 100 for manuscript uncertainty summaries.")
    return _run_single_track(args)


if __name__ == "__main__":
    raise SystemExit(main())
