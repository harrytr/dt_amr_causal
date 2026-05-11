#!/usr/bin/env python3
from __future__ import annotations

"""
counterfactual_rollout.py

Stage 1 paired factual/counterfactual rollout runner.

This wrapper drives generate_amr_data.py twice, optionally converts both outputs
to .pt, and then computes simulator-world causal estimands and polished figures.

Compatibility notes
-------------------
- Supports both the legacy single-intervention flag (--intervention_json) and
  the newer split-arm flags (--factual_intervention_json and
  --counterfactual_intervention_json).
- generate_amr_data.py expects --causal_intervention_json to be a JSON string,
  not a filesystem path. This wrapper therefore loads intervention files and
  forwards the serialized JSON payload.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from causal_interventions import describe_intervention, load_intervention_json


REPO_DIR = Path(__file__).resolve().parent
GENERATE_SCRIPT = REPO_DIR / "generate_amr_data.py"
CONVERT_SCRIPT = REPO_DIR / "convert_to_pt.py"
ESTIMANDS_SCRIPT = REPO_DIR / "causal_estimands.py"


class RolloutError(RuntimeError):
    pass


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONUTF8"] = "1"
    merged_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RolloutError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stage_dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_intervention_path(path_value: str) -> Optional[Path]:
    s = str(path_value).strip()
    if not s:
        return None
    return Path(s).resolve()


def _build_generate_cmd(
    *,
    python_bin: str,
    output_dir: Path,
    seed: int,
    pair_id: str,
    role: str,
    shared_noise_seed: int,
    intervention_payload_json: Optional[str],
    passthrough_args: List[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(GENERATE_SCRIPT),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--causal_mode",
        "1",
        "--causal_pair_id",
        pair_id,
        "--causal_role",
        role,
        "--causal_shared_noise_seed",
        str(shared_noise_seed),
    ]
    if intervention_payload_json is not None:
        cmd.extend(["--causal_intervention_json", intervention_payload_json])
    cmd.extend(passthrough_args)
    return cmd


def _build_convert_cmd(
    *,
    python_bin: str,
    graphml_dir: Path,
    horizons: str,
    state_mode: str,
) -> List[str]:
    return [
        python_bin,
        str(CONVERT_SCRIPT),
        "--graphml_dir",
        str(graphml_dir),
        "--horizons",
        horizons,
        "--state_mode",
        state_mode,
        "--keep_graphml",
    ]


def _build_estimands_cmd(
    *,
    python_bin: str,
    factual_dir: Path,
    counterfactual_dir: Path,
    out_dir: Path,
    stem: str,
) -> List[str]:
    return [
        python_bin,
        str(ESTIMANDS_SCRIPT),
        "--factual_dir",
        str(factual_dir),
        "--counterfactual_dir",
        str(counterfactual_dir),
        "--out_dir",
        str(out_dir),
        "--stem",
        stem,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True, type=str)
    parser.add_argument("--intervention_json", type=str, default="")
    parser.add_argument("--factual_intervention_json", type=str, default="")
    parser.add_argument("--counterfactual_intervention_json", type=str, default="")
    parser.add_argument(
        "--state_mode",
        type=str,
        default=os.environ.get("DT_STATE_MODE", "ground_truth"),
        choices=["ground_truth", "partial_observation"],
    )
    parser.add_argument("--pair_id", type=str, default="")
    parser.add_argument("--shared_noise_seed", type=int, default=20260401)
    parser.add_argument("--factual_seed", type=int, default=4100)
    parser.add_argument("--counterfactual_seed", type=int, default=4100)
    parser.add_argument("--horizons", type=str, default="7,14")
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--stem", type=str, default="causal")
    parser.add_argument(
        "sim_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed through directly to generate_amr_data.py. "
            "Prefix with --, e.g. -- --num_days 60 --num_patients 200"
        ),
    )
    args = parser.parse_args()

    python_bin = sys.executable
    out_root = Path(args.out_root).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_id = args.pair_id.strip() or f"cfpair_{uuid.uuid4().hex[:12]}"

    factual_path = _resolve_intervention_path(args.factual_intervention_json)
    counterfactual_path = _resolve_intervention_path(args.counterfactual_intervention_json)

    if counterfactual_path is None:
        legacy_path = _resolve_intervention_path(args.intervention_json)
        if legacy_path is not None:
            counterfactual_path = legacy_path

    factual_spec = load_intervention_json(factual_path) if factual_path is not None else None
    counterfactual_spec = load_intervention_json(counterfactual_path) if counterfactual_path is not None else None

    if factual_spec is not None:
        _write_json(out_root / "factual_intervention.json", factual_spec.to_dict())
    if counterfactual_spec is not None:
        _write_json(out_root / "counterfactual_intervention.json", counterfactual_spec.to_dict())

    factual_dir = _stage_dir(out_root, "factual_graphml")
    counterfactual_dir = _stage_dir(out_root, "counterfactual_graphml")
    estimands_dir = _stage_dir(out_root, "causal_estimands")

    passthrough_args = list(args.sim_args)
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    factual_payload_json = None if factual_spec is None else json.dumps(factual_spec.to_dict(), sort_keys=True)
    counterfactual_payload_json = None if counterfactual_spec is None else json.dumps(counterfactual_spec.to_dict(), sort_keys=True)

    factual_cmd = _build_generate_cmd(
        python_bin=python_bin,
        output_dir=factual_dir,
        seed=int(args.factual_seed),
        pair_id=pair_id,
        role="factual",
        shared_noise_seed=int(args.shared_noise_seed),
        intervention_payload_json=factual_payload_json,
        passthrough_args=passthrough_args,
    )
    counterfactual_cmd = _build_generate_cmd(
        python_bin=python_bin,
        output_dir=counterfactual_dir,
        seed=int(args.counterfactual_seed),
        pair_id=pair_id,
        role="counterfactual",
        shared_noise_seed=int(args.shared_noise_seed),
        intervention_payload_json=counterfactual_payload_json,
        passthrough_args=passthrough_args,
    )

    _run_cmd(factual_cmd)
    _run_cmd(counterfactual_cmd)

    if not args.skip_convert:
        _run_cmd(
            _build_convert_cmd(
                python_bin=python_bin,
                graphml_dir=factual_dir,
                horizons=args.horizons,
                state_mode=args.state_mode,
            )
        )
        _run_cmd(
            _build_convert_cmd(
                python_bin=python_bin,
                graphml_dir=counterfactual_dir,
                horizons=args.horizons,
                state_mode=args.state_mode,
            )
        )

    _run_cmd(
        _build_estimands_cmd(
            python_bin=python_bin,
            factual_dir=factual_dir,
            counterfactual_dir=counterfactual_dir,
            out_dir=estimands_dir,
            stem=args.stem,
        )
    )

    summary = {
        "pair_id": pair_id,
        "state_mode": args.state_mode,
        "shared_noise_seed": int(args.shared_noise_seed),
        "factual_seed": int(args.factual_seed),
        "counterfactual_seed": int(args.counterfactual_seed),
        "factual_intervention": None if factual_spec is None else factual_spec.to_dict(),
        "counterfactual_intervention": None if counterfactual_spec is None else counterfactual_spec.to_dict(),
        "factual_intervention_description": None if factual_spec is None else describe_intervention(factual_spec),
        "counterfactual_intervention_description": None if counterfactual_spec is None else describe_intervention(counterfactual_spec),
        "factual_dir": str(factual_dir),
        "counterfactual_dir": str(counterfactual_dir),
        "estimands_dir": str(estimands_dir),
        "converted_to_pt": not bool(args.skip_convert),
        "horizons": str(args.horizons),
        "passthrough_sim_args": passthrough_args,
    }
    _write_json(out_root / "rollout_manifest.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())