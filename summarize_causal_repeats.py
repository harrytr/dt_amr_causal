#!/usr/bin/env python3
"""Aggregate paired causal policy repeats without pooling decision windows.

Uses the existing evaluator's metric definitions, including oracle ties. The
primary interval is a Student-t interval across independent joint repetitions.
Within-fit test intervals resample whole simulation seeds, not individual rows.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import warnings
import numpy as np
from scipy.stats import t

from causal_repeats import atomic_json, digest, file_hash, read_csv, read_json, write_csv


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def number(value):
    return float(value) if value is not None else float("nan")


def policy_metrics(states, actions):
    from evaluate_policy_selector import (_multiclass_state_metrics,
        _multiclass_state_metrics_tie_aware, _classification_row_auc)

    def mean(key):
        values = [float(r[key]) for r in states if str(r.get(key, "")).strip()]
        return float(np.mean(values)) if values else float("nan")

    result = {"policy_accuracy": mean("policy_match"), "strict_policy_accuracy": mean("strict_policy_match"),
              "top2_accuracy": mean("top2_match"), "strict_top2_accuracy": mean("strict_top2_match"),
              "mean_regret": mean("regret"), "median_regret": float(np.median([float(r["regret"]) for r in states])),
              "mean_baseline_oracle_delta": mean("baseline_oracle_delta"),
              "baseline_improvement_rate": mean("baseline_improved"), "baseline_tie_rate": mean("baseline_tie"),
              "baseline_non_worse_rate": float(np.mean([int(r["baseline_improved"]) or int(r["baseline_tie"]) for r in states])),
              "tie_state_rate": float(np.mean([int(r["oracle_tie_count"]) > 1 for r in states]))}
    for prefix, function in (("strict_", _multiclass_state_metrics), ("tie_aware_", _multiclass_state_metrics_tie_aware)):
        result.update({prefix + name: float(value) for name, value in function(states).items()})
    result["action_row_auroc"] = float(_classification_row_auc(actions, states)["auroc"])
    return result


def check_points(metrics, original):
    reference = {key: original[key] for key in ("policy_accuracy", "strict_policy_accuracy", "top2_accuracy",
        "strict_top2_accuracy", "baseline_improvement_rate", "baseline_tie_rate", "baseline_non_worse_rate", "tie_state_rate")}
    reference.update(mean_regret=original["regret"]["mean"], median_regret=original["regret"]["median"],
                     action_row_auroc=original["classification_metrics"]["auroc"])
    for prefix, key in (("strict_", "classification_metrics"), ("tie_aware_", "tie_aware_classification_metrics")):
        reference.update({prefix + k: v for k, v in original[key].items() if k != "auroc"})
    for key, point in reference.items():
        a, b = number(point), number(metrics[key])
        if not ((not math.isfinite(a) and not math.isfinite(b)) or math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10)):
            raise RuntimeError(f"Aggregation changed the evaluator's {key}: {a} versus {b}")


def mean_interval(values, planned):
    vals = np.asarray([number(v) for v in values], dtype=float)
    count = int(np.isfinite(vals).sum())
    result = {"n_planned": planned, "n_completed": len(vals), "n_defined": count,
              "mean": None, "sd": None, "ci95_low": None, "ci95_high": None,
              "interval_method": "Student-t across independent joint repeats"}
    if count != len(vals) or count == 0:
        result["status"] = "undefined_metric; interval withheld"
        return result
    result["mean"] = float(np.mean(vals))
    if count >= 2:
        result["sd"] = float(np.std(vals, ddof=1))
    if count != planned or count < 2:
        result["status"] = "incomplete_design; interval withheld"
        return result
    half = float(t.ppf(.975, count-1) * result["sd"] / math.sqrt(count))
    result.update(ci95_low=result["mean"]-half, ci95_high=result["mean"]+half, status="complete")
    return result


def conditional_intervals(states, actions, cfg, seed):
    clusters = sorted({int(r["seed"]) for r in states})
    state_groups = {s: [r for r in states if int(r["seed"]) == s] for s in clusters}
    action_groups = {s: [r for r in actions if int(r["seed"]) == s] for s in clusters}
    point = policy_metrics(states, actions)
    boot = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    if len(clusters) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(cfg["bootstrap_replicates"]):
                sample = rng.choice(clusters, size=len(clusters), replace=True)
                ms = policy_metrics([r for s in sample for r in state_groups[int(s)]],
                                    [r for s in sample for r in action_groups[int(s)]])
                for name, value in ms.items():
                    boot[name].append(value)
    result = []
    for name, value in point.items():
        samples = np.asarray(boot[name], dtype=float)
        finite = samples[np.isfinite(samples)]
        ready = len(clusters) >= 2 and len(finite) == cfg["bootstrap_replicates"] and math.isfinite(value)
        low, high = np.quantile(finite, [.025, .975]) if ready else (None, None)
        result.append({"metric": name, "point": value, "ci95_low": low, "ci95_high": high,
                       "n_simulation_seeds": len(clusters), "n_bootstrap": len(samples),
                       "n_defined_bootstrap": len(finite), "status": "complete" if ready else "insufficient_clusters_or_undefined_resamples; interval withheld",
                       "interval_method": "whole-test-simulation-seed percentile bootstrap; conditional on this fitted model and parameter draw"})
    return clean(result)


def checked_stage(folder, files):
    marker = folder / "complete.json"
    if not marker.is_file():
        raise RuntimeError(f"Incomplete stage in a completed repetition: {folder}")
    meta = read_json(marker)
    for name in files:
        if not (folder / name).is_file() or file_hash(folder / name) != meta["files"].get(name):
            raise RuntimeError(f"Changed/missing result: {folder / name}")


def latex_table(rows, path, planned):
    lookup = {(r["track"], r["split"], r["metric"]): r for r in rows}
    metrics = ["action_row_auroc", "policy_accuracy", "tie_aware_macro_f1", "tie_aware_macro_sensitivity",
               "tie_aware_macro_specificity", "mean_regret"]
    labels = {"ground_truth": "Ground truth", "partial_observation": "Partial observation"}
    splits = {"train": "Train", "validation": "Validation", "test": "Test"}
    def cell(row):
        if row is None or row["mean"] is None:
            return "--"
        point = f"{row['mean']:.3f}"
        if row["ci95_low"] is None:
            return point + r"$^{\dagger}$"
        return r"\shortstack{" + point + r"\\(" + f"{row['ci95_low']:.3f}, {row['ci95_high']:.3f}" + ")}"
    lines = [r"% Requires booktabs and adjustbox. Means and 95% intervals are across joint repeats.",
             r"\begin{table*}[!htbp]", r"\centering\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Observation & Split & Action AUROC & Policy accuracy & Macro F1 & Sensitivity & Specificity & Mean regret \\", r"\midrule"]
    for track in labels:
        if not any(k[0] == track for k in lookup):
            continue
        for split in splits:
            lines.append(" & ".join([labels[track], splits[split]] + [cell(lookup.get((track, split, metric))) for metric in metrics]) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\caption{Repeated causal policy-selection experiment at $T=7$, $H=7$. " +
              f"The design contains {planned} independent joint repetitions of simulation seeds, model initialization and exploratory parameter draws. " +
              r"Entries are repeat means with 95\% Student-$t$ confidence intervals in parentheses. " +
              r"Accuracy and macro metrics retain the original tie-aware definitions. Sensitivity and specificity are macro one-versus-rest metrics. " +
              r"Action AUROC distinguishes oracle-optimal from non-optimal state--action rows; it is not the predictive experiment's binary AUROC. " +
              r"Regret uses the original seven-day transmission--importation resistant-gain target. " +
              r"$\dagger$ denotes an incomplete design with its interval withheld; -- denotes an undefined/unavailable metric. " +
              r"Intervals are not clipped to metric bounds.}", r"\label{tab:causal-policy-repeats}", r"\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(output):
    output = Path(output).resolve()
    frozen = read_json(output / "run_manifest.json")["identity"]
    cfg, design = frozen["config"], frozen["design"]
    planned = len(design["repeats"])
    target = output / "summary"
    target.mkdir(exist_ok=True)
    per_repeat, conditional, milp_rows, complete_ids, input_hashes = [], [], [], [], []
    for rep in design["repeats"]:
        r = rep["repeat"]
        root = output / f"repeat_{r:03d}"
        if not (root / "repeat_complete.json").is_file():
            continue
        complete_ids.append(r)
        track_targets = []
        for track in cfg["tracks"]:
            work = root / f"TRACK_{track}" / "work"
            for split in ("train", "validation", "test"):
                ev = work / f"policy_eval_{split}"
                files = ["policy_selection_summary.json", "policy_selection_per_state_requested_split.csv", "policy_action_scores_requested_split.csv"]
                checked_stage(ev, files)
                original = read_json(ev / files[0])
                states, actions = read_csv(ev / files[1]), read_csv(ev / files[2])
                expected = {(s, d) for s in rep["simulator_seeds"][split] for d in rep["decision_days"]}
                if len(states) != len(expected) or {(int(s["seed"]), int(s["decision_day"])) for s in states} != expected:
                    raise RuntimeError(f"State population changed in {ev}")
                identity = sorted((s["state_id"], s["seed"], s["decision_day"], s["oracle_best_action_id"], s["oracle_tied_best_action_ids_json"]) for s in states)
                track_targets.append((split, digest(identity)))
                metrics = policy_metrics(states, actions)
                check_points(metrics, original)
                per_repeat.append({"repeat": r, "track": track, "split": split,
                    "n_simulation_seeds": len(rep["simulator_seeds"][split]), "n_states": len(states),
                    "model_seed": rep["model_seed"], **rep["parameter_multipliers"], **clean(metrics)})
                if split == "test":
                    ci_key = digest({"files": {n: file_hash(ev/n) for n in files}, "bootstrap": cfg["bootstrap_replicates"],
                                     "seed": cfg["bootstrap_seed"]+r, "summary_source": file_hash(Path(__file__))})
                    ci_path = target / "conditional_cache" / f"repeat_{r:03d}_{track}.json"
                    if ci_path.exists() and read_json(ci_path).get("key") == ci_key:
                        ci = read_json(ci_path)["rows"]
                    else:
                        ci = conditional_intervals(states, actions, cfg, cfg["bootstrap_seed"]+r)
                        atomic_json(ci_path, {"key": ci_key, "rows": ci})
                    conditional.extend({"repeat": r, "track": track, "split": split, **row} for row in ci)
                input_hashes.append({"repeat": r, "track": track, "split": split,
                                     "files": {n: file_hash(ev/n) for n in files}})
            if cfg["run_milp"]:
                milp = work / rep["milp"]["out_name"]
                checked_stage(milp, ["milp_policy_summary.json", "milp_policy_plan.csv"])
                sm = read_json(milp / "milp_policy_summary.json")
                plans = read_csv(milp / "milp_policy_plan.csv")
                n = len(rep["simulator_seeds"]["test"]) * len(rep["decision_days"])
                if len(plans) != n or sm["n_states"] != n:
                    raise RuntimeError(f"MILP omitted test states: {milp}")
                milp_rows.append({"repeat": r, "track": track, "n_states": n,
                    "solver": sm["solver_name"], "solver_status": sm["solver_status"],
                    "termination_condition": sm["termination_condition"],
                    "predicted_objective": sm["objective_value"], "total_cost": sm["total_cost"],
                    "total_predicted_gain_vs_baseline": sm["total_gain_vs_baseline"],
                    "mean_predicted_gain_vs_baseline": float(sm["total_gain_vs_baseline"])/n})
        for split in ("train", "validation", "test"):
            if len({h for s, h in track_targets if s == split}) != 1:
                raise RuntimeError(f"Tracks do not use matched causal outcomes in repeat {r}, {split}")
    if not per_repeat:
        atomic_json(target / "status.json", {"planned_repeats": planned, "completed_repeats": [], "status": "no_complete_repeats"})
        return
    metric_names = [k for k in per_repeat[0] if k not in {"repeat", "track", "split", "n_simulation_seeds", "n_states", "model_seed", *cfg["parameter_multipliers"]}]
    aggregate = []
    for track in cfg["tracks"]:
        for split in ("train", "validation", "test"):
            rows = [r for r in per_repeat if r["track"] == track and r["split"] == split]
            for metric in metric_names:
                aggregate.append({"track": track, "split": split, "metric": metric,
                                  **mean_interval([r[metric] for r in rows], planned)})
    paired = []
    if set(cfg["tracks"]) == {"ground_truth", "partial_observation"}:
        lookup = {(r["repeat"], r["track"], r["split"]): r for r in per_repeat}
        for split in ("train", "validation", "test"):
            for metric in metric_names:
                values = [number(lookup[(r, "ground_truth", split)][metric]) - number(lookup[(r, "partial_observation", split)][metric]) for r in complete_ids]
                paired.append({"comparison": "ground_truth_minus_partial_observation", "split": split,
                               "metric": metric, **mean_interval(values, planned)})
    milp_aggregate = []
    for track in cfg["tracks"]:
        for metric in ("predicted_objective", "total_cost", "total_predicted_gain_vs_baseline", "mean_predicted_gain_vs_baseline"):
            values = [r[metric] for r in milp_rows if r["track"] == track]
            if values:
                milp_aggregate.append({"track": track, "metric": metric, **mean_interval(values, planned)})
    for name, rows in (("policy_per_repeat.csv", per_repeat), ("policy_mean_ci95.csv", aggregate),
                       ("policy_paired_track_difference_ci95.csv", paired), ("policy_test_conditional_ci95.csv", conditional),
                       ("milp_per_repeat.csv", milp_rows), ("milp_mean_ci95.csv", milp_aggregate)):
        write_csv(target / name, rows)
    latex_table(aggregate, target / "causal_policy_repeats.tex", planned)
    atomic_json(target / "summary.json", clean({"planned_repeats": planned, "completed_repeats": complete_ids,
        "policy_mean_ci95": aggregate, "paired_track_difference_ci95": paired, "milp_mean_ci95": milp_aggregate,
        "input_hashes": input_hashes, "parameter_rationale": cfg["parameter_rationale"],
        "inference_scope": "Joint variation in simulator seeds, model initialization and parameter draws; variance components are not separated. Approximate Student-t mean intervals across repeats, with no window-level pseudo-replication. Conditional test intervals resample whole simulator trajectories within a fixed fit/draw.",
        "milp_scope": "Objectives and gain values are predicted-score quantities, not realized clinical effects. Single-decision counterfactual outcomes do not establish the effect of executing an entire sequence of MILP actions."}))
    print(f"[summary] {len(complete_ids)}/{planned} complete repeats; {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat_output", default="causal_repeat_results")
    args = parser.parse_args()
    from causal_repeats import lock
    with lock(Path(args.repeat_output) / "run.lock"):
        summarize(args.repeat_output)


if __name__ == "__main__":
    main()
