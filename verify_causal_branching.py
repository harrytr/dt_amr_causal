#!/usr/bin/env python3
from __future__ import annotations

"""
verify_causal_branching.py
==========================


python verify_causal_branching.py \
  --manifest_csv experiments_causal_results/TRACK_ground_truth/work/causal_policy_temporal_heterogeneity/policy_manifest.csv \
  --max_states 10
  
  
Sanity check for causal policy datasets built by build_causal_policy_dataset.py.

What it verifies
----------------
For each chosen state_id in the policy manifest, this script checks that:

1. All actions for that state share the exact same past input window
   (window_graphml_json and window_pt_json).

2. The decision-day graph is identical across actions.

3. Non-baseline actions diverge only after action_start_day.
   In particular, for each non-baseline action:
   - days < action_start_day must match baseline exactly
   - at least one day >= action_start_day should differ from baseline
     (unless the intervention truly has no effect in that run)

This gives a concrete scientific sanity check that the dataset builder is
respecting the intended counterfactual semantics:
same past up to decision day, different future after intervention starts.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class VerificationError(RuntimeError):
    """Raised when verification fails."""


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest_rows(manifest_csv: Path) -> List[Dict[str, Any]]:
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    rows: List[Dict[str, Any]] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise VerificationError(f"Manifest CSV is empty: {manifest_csv}")
    return rows


def _parse_json_list_field(value: str, field_name: str, row_hint: str) -> List[Path]:
    s = str(value or "").strip()
    if s == "":
        raise VerificationError(f"Missing {field_name} for row {row_hint}")
    try:
        payload = json.loads(s)
    except Exception as exc:
        raise VerificationError(f"Invalid JSON in {field_name} for row {row_hint}: {exc}") from exc
    if not isinstance(payload, list):
        raise VerificationError(f"{field_name} is not a list for row {row_hint}")
    out: List[Path] = []
    for item in payload:
        p = Path(str(item))
        out.append(p)
    return out


def _group_rows_by_state(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        state_id = str(row.get("state_id", "")).strip()
        if state_id == "":
            raise VerificationError("Found row with empty state_id")
        grouped.setdefault(state_id, []).append(row)
    return grouped


def _find_baseline_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("is_baseline", "")).strip() == "1":
            return row
    raise VerificationError("No baseline row found for state")


def _load_window_paths(row: Dict[str, Any]) -> Tuple[List[Path], List[Path]]:
    row_hint = str(row.get("pair_id", row.get("state_id", "unknown")))
    graphml_paths = _parse_json_list_field(str(row.get("window_graphml_json", "")), "window_graphml_json", row_hint)
    pt_paths = _parse_json_list_field(str(row.get("window_pt_json", "")), "window_pt_json", row_hint)
    return graphml_paths, pt_paths


def _check_shared_windows(state_id: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []

    first_graphml: Optional[List[str]] = None
    first_pt: Optional[List[str]] = None

    for row in rows:
        graphml_paths, pt_paths = _load_window_paths(row)
        graphml_norm = [str(p.resolve()) for p in graphml_paths]
        pt_norm = [str(p.resolve()) for p in pt_paths]

        if first_graphml is None:
            first_graphml = graphml_norm
            first_pt = pt_norm
            continue

        if graphml_norm != first_graphml:
            issues.append(
                f"[{state_id}] window_graphml_json differs across actions "
                f"(pair_id={row.get('pair_id', '')})"
            )
        if pt_norm != first_pt:
            issues.append(
                f"[{state_id}] window_pt_json differs across actions "
                f"(pair_id={row.get('pair_id', '')})"
            )

    return issues


def _check_decision_day_identity(state_id: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    first_hash: Optional[str] = None
    first_path: Optional[Path] = None

    for row in rows:
        path = Path(str(row.get("decision_graphml_path", "")))
        if not path.is_file():
            issues.append(f"[{state_id}] decision_graphml_path missing: {path}")
            continue
        digest = _sha256_file(path)
        if first_hash is None:
            first_hash = digest
            first_path = path
            continue
        if digest != first_hash:
            issues.append(
                f"[{state_id}] decision-day graph differs across actions: "
                f"{first_path} vs {path}"
            )

    return issues


def _index_graphml_days(graphml_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for path in sorted(graphml_dir.glob("*.graphml")):
        name = path.name
        marker = "_t"
        idx = name.rfind(marker)
        if idx < 0:
            continue
        j = idx + len(marker)
        digits: List[str] = []
        while j < len(name) and name[j].isdigit():
            digits.append(name[j])
            j += 1
        if not digits:
            continue
        day = int("".join(digits))
        out[day] = path
    return out


def _compare_file_hash(a: Path, b: Path) -> bool:
    return _sha256_file(a) == _sha256_file(b)


def _check_branching_against_baseline(
    state_id: str,
    baseline_row: Dict[str, Any],
    action_row: Dict[str, Any],
    require_future_difference: bool,
) -> List[str]:
    issues: List[str] = []

    baseline_dir = Path(str(baseline_row.get("action_graphml_dir", "")))
    action_dir = Path(str(action_row.get("action_graphml_dir", "")))

    if not baseline_dir.is_dir():
        issues.append(f"[{state_id}] baseline action_graphml_dir missing: {baseline_dir}")
        return issues
    if not action_dir.is_dir():
        issues.append(
            f"[{state_id}] action graphml dir missing for action {action_row.get('action_id', '')}: {action_dir}"
        )
        return issues

    baseline_days = _index_graphml_days(baseline_dir)
    action_days = _index_graphml_days(action_dir)

    decision_day = int(str(action_row.get("decision_day", "0")))
    action_start_day = int(str(action_row.get("action_start_day", "0")))

    if action_start_day <= 0:
        issues.append(
            f"[{state_id}] invalid action_start_day={action_start_day} "
            f"for action {action_row.get('action_id', '')}"
        )
        return issues

    # Days before action_start_day must match baseline exactly.
    for day in range(1, action_start_day):
        bp = baseline_days.get(day)
        ap = action_days.get(day)
        if bp is None or ap is None:
            issues.append(
                f"[{state_id}] missing pre-start day {day} for action {action_row.get('action_id', '')}"
            )
            continue
        if not _compare_file_hash(bp, ap):
            issues.append(
                f"[{state_id}] pre-start divergence detected for action {action_row.get('action_id', '')} "
                f"on day {day} (decision_day={decision_day}, action_start_day={action_start_day})"
            )

    # We expect future divergence at or after action_start_day.
    first_diff_day: Optional[int] = None
    common_future_days = sorted(set(d for d in baseline_days if d >= action_start_day) & set(d for d in action_days if d >= action_start_day))
    for day in common_future_days:
        if not _compare_file_hash(baseline_days[day], action_days[day]):
            first_diff_day = day
            break

    if require_future_difference and first_diff_day is None:
        issues.append(
            f"[{state_id}] no post-start divergence detected for action {action_row.get('action_id', '')} "
            f"(action_start_day={action_start_day})"
        )

    return issues


def _choose_state_ids(
    grouped: Dict[str, List[Dict[str, Any]]],
    state_ids: Sequence[str],
    max_states: int,
) -> List[str]:
    if state_ids:
        missing = [sid for sid in state_ids if sid not in grouped]
        if missing:
            raise VerificationError(f"Requested state_id(s) not found: {missing}")
        return list(state_ids)
    all_ids = sorted(grouped.keys())
    return all_ids[: max_states]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify causal branching semantics in a policy manifest dataset.")
    parser.add_argument("--manifest_csv", required=True, type=str, help="Path to policy_manifest.csv")
    parser.add_argument(
        "--state_ids",
        type=str,
        default="",
        help="Comma-separated specific state_id values to verify. If empty, verifies the first N states.",
    )
    parser.add_argument(
        "--max_states",
        type=int,
        default=10,
        help="Number of states to verify when --state_ids is not provided.",
    )
    parser.add_argument(
        "--allow_no_future_difference",
        action="store_true",
        help="Allow actions whose future never diverges from baseline in the checked horizon.",
    )
    args = parser.parse_args()

    manifest_csv = Path(args.manifest_csv).resolve()
    rows = _load_manifest_rows(manifest_csv)
    grouped = _group_rows_by_state(rows)

    requested_state_ids = [x.strip() for x in str(args.state_ids).split(",") if x.strip() != ""]
    chosen_state_ids = _choose_state_ids(grouped, requested_state_ids, max(1, int(args.max_states)))

    all_issues: List[str] = []
    checked_actions = 0

    for state_id in chosen_state_ids:
        state_rows = grouped[state_id]
        baseline_row = _find_baseline_row(state_rows)

        all_issues.extend(_check_shared_windows(state_id, state_rows))
        all_issues.extend(_check_decision_day_identity(state_id, state_rows))

        for row in state_rows:
            if str(row.get("is_baseline", "")).strip() == "1":
                continue
            checked_actions += 1
            all_issues.extend(
                _check_branching_against_baseline(
                    state_id=state_id,
                    baseline_row=baseline_row,
                    action_row=row,
                    require_future_difference=not bool(args.allow_no_future_difference),
                )
            )

    print(f"Checked states: {len(chosen_state_ids)}")
    print(f"Checked non-baseline actions: {checked_actions}")

    if all_issues:
        print("\nVERIFICATION FAILED\n")
        for issue in all_issues:
            print(issue)
        return 1

    print("\nVERIFICATION PASSED")
    print("All checked states share identical past windows and decision-day graphs across actions,")
    print("and no action diverges before its action_start_day.")
    if not bool(args.allow_no_future_difference):
        print("At least one post-start divergence was detected for every checked non-baseline action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
