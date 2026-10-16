"""Verified, resumable arm cache for the opt-in causal repeat driver.

Rollouts are shared between observation tracks. Ordinary dataset builds never
enter this module. A partial collection of PT files is not treated as complete.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from causal_repeats import atomic_json, digest, file_hash, lock, read_json

DAY = re.compile(r"_t(\d+)(?:_L[^.]*)?\.(?:pt|graphml)$")


def indexed(folder, suffix):
    result = {}
    for path in sorted(Path(folder).glob("*" + suffix)):
        match = DAY.search(path.name)
        if not match or int(match[1]) in result:
            raise RuntimeError(f"Unrecognized/duplicate day file: {path}")
        result[int(match[1])] = path
    return result


def describe_files(folder, suffix=".pt"):
    return {p.name: {"size": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(Path(folder).glob("*" + suffix))}


def validate_marker(folder, name="complete.json"):
    folder = Path(folder)
    try:
        payload = read_json(folder / name)
        files = payload["files"]
        return bool(files) and set(files) == {p.name for p in folder.glob("*.pt")} and all(
            (folder / n).is_file() and (folder / n).stat().st_size == meta["size"]
            and file_hash(folder / n) == meta["sha256"] for n, meta in files.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)
    # Local caches and track datasets share immutable PT files when possible.
    try:
        os.link(source, tmp)
    except OSError:
        shutil.copy2(source, tmp)
    os.replace(tmp, target)


def install_mode(source, target, marker="complete.json"):
    payload = read_json(source / "complete.json")
    target.mkdir(parents=True, exist_ok=True)
    (target / marker).unlink(missing_ok=True)
    for old in target.glob("*.pt"):
        if old.name not in payload["files"]:
            old.unlink()
    for name in payload["files"]:
        copy_file(source / name, target / name)
    atomic_json(target / marker, payload)


def graph_signature(path):
    """Compare simulator state, ignoring only arm provenance in graph metadata."""
    import networkx as nx
    g = nx.read_graphml(path)
    graph = {k: v for k, v in g.graph.items() if not str(k).startswith("cf_")}
    nodes = sorted((str(n), dict(a)) for n, a in g.nodes(data=True))
    edges = []
    if g.is_multigraph():
        for u, v, key, data in g.edges(keys=True, data=True):
            ends = (str(u), str(v)) if g.is_directed() else tuple(sorted((str(u), str(v))))
            edges.append((*ends, str(key), dict(data)))
    else:
        for u, v, data in g.edges(data=True):
            ends = (str(u), str(v)) if g.is_directed() else tuple(sorted((str(u), str(v))))
            edges.append((*ends, dict(data)))
    edges.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return digest({"graph": graph, "nodes": nodes, "edges": edges,
                   "directed": g.is_directed(), "multigraph": g.is_multigraph()})


def cache_roots(root):
    root = Path(root)
    return [root / "arm_cache", root]


def plan_cache_status(output, roots):
    return {"local_mode_markers": len(list((Path(output) / "arm_cache").glob("*/TRACK_*/complete.json"))),
            "reuse_roots": [str(r) for r in roots],
            "note": "Markers are only an inventory; every reused arm must match its scientific identity and file checksums."}


def prepare_arm_cached(*, python_bin, out_dir, seed, pair_id, role, shared_noise_seed,
                       intervention_json, sim_args, horizons, state_mode, force):
    from build_causal_policy_dataset import _build_generate_cmd, _build_convert_cmd, REPO_DIR
    if force:
        raise ValueError("Repeat mode must use force=false; change the output directory for a new design")
    context = read_json(os.environ["DT_CAUSAL_REPEAT_CONTEXT"])
    tracks = context["tracks"]
    if state_mode not in tracks:
        raise ValueError("Requested observation track is absent from the frozen cache context")
    sim_args = list(map(str, sim_args))
    days = int(sim_args[sim_args.index("--num_days") + 1])
    intervention = json.loads(intervention_json) if str(intervention_json).strip() else None
    identity = {"cache_version": 1, "seed": int(seed), "pair_id": pair_id, "role": role,
                "shared_noise_seed": int(shared_noise_seed), "intervention": intervention,
                "sim_args": sim_args, "horizons": str(horizons),
                "sources": {k: context["sources"][k] for k in
                            ("generate_amr_data.py", "causal_interventions.py", "convert_to_pt.py")}}
    key = digest(identity)
    base_identity = {k: v for k, v in identity.items() if k not in {"pair_id", "role", "intervention"}}
    base_identity["baseline_intervention"] = context["baseline_intervention"]
    root = Path(context["cache_root"])
    arm = root / key
    baseline_index = root / "baseline_index" / (digest(base_identity) + ".json")
    decision_day = None
    if role == "counterfactual":
        decision_day = int(intervention["start_day"]) - 1
        if decision_day < 7 or decision_day + 7 > days:
            raise ValueError("Counterfactual arm lacks the full history or target horizon")
    wanted = set(range(1, days + 1)) if decision_day is None or context["keep_action_full_pt"] else {decision_day}
    with lock(arm / "arm.lock", wait=True):
        identity_file = arm / "identity.json"
        if identity_file.exists() and read_json(identity_file) != identity:
            raise RuntimeError(f"Cache identity mismatch: {arm}")
        atomic_json(identity_file, identity)
        # Borrow only verified repeat-cache entries; legacy outputs without
        # parameter/seed provenance cannot establish matching counterfactuals.
        for mode in tracks:
            dest = arm / f"TRACK_{mode}"
            if validate_marker(dest) and set(indexed(dest, ".pt")) >= wanted:
                continue
            for previous in context["reuse_roots"]:
                found = False
                for oldroot in cache_roots(previous):
                    source_arm = oldroot / key
                    source = source_arm / f"TRACK_{mode}"
                    if (source_arm / "identity.json").is_file() and read_json(source_arm / "identity.json") == identity and validate_marker(source) and set(indexed(source, ".pt")) >= wanted:
                        install_mode(source, dest)
                        for name in ("raw_complete.json", "branch_audit.json"):
                            if (source_arm / name).is_file():
                                shutil.copy2(source_arm / name, arm / name)
                        found = True
                        break
                if found:
                    break
        all_ready = all(validate_marker(arm / f"TRACK_{mode}") and
                        set(indexed(arm / f"TRACK_{mode}", ".pt")) >= wanted for mode in tracks)
        if all_ready and not (arm / "branch_audit.json").is_file():
            all_ready = False
        if not all_ready:
            raw = arm / "raw"
            raw.mkdir(exist_ok=True)
            raw_marker = arm / "raw_complete.json"
            raw_ready = False
            if raw_marker.is_file():
                meta = read_json(raw_marker)
                raw_ready = (meta.get("identity_sha256") == key and set(indexed(raw, ".graphml")) == set(range(1, days+1))
                             and all((raw / n).is_file() and file_hash(raw / n) == h for n, h in meta["files"].items()))
            env = os.environ.copy()
            for variable in ("DT_PT_OUT_DIR", "DT_STATE_MODE", "DT_SIM_EXTRA_ARGS", "DT_CONVERT_EXTRA_ARGS"):
                env.pop(variable, None)
            env["PYTHONHASHSEED"] = str(seed)
            env["MPLBACKEND"] = "Agg"

            def execute(command, log_name):
                with (arm / log_name).open("a", encoding="utf-8") as log:
                    status = subprocess.run(command, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
                if status.returncode:
                    raise RuntimeError(f"Arm stage failed: {arm / log_name}")

            if not raw_ready:
                shutil.rmtree(raw)
                raw.mkdir()
                raw_marker.unlink(missing_ok=True)
                print(f"[arm] generate seed={seed} role={role} decision_day={decision_day}", flush=True)
                execute(_build_generate_cmd(python_bin=python_bin, output_dir=raw, seed=seed,
                            pair_id=pair_id, role=role, shared_noise_seed=shared_noise_seed,
                            intervention_json=intervention_json, sim_args=sim_args), "generate.log")
                graphs = indexed(raw, ".graphml")
                if set(graphs) != set(range(1, days + 1)):
                    raise RuntimeError(f"Incomplete rollout: {raw}")
                signatures = {str(d): graph_signature(p) for d, p in graphs.items()}
                atomic_json(raw_marker, {"identity_sha256": key, "signatures": signatures,
                            "files": {p.name: file_hash(p) for p in graphs.values()}})
            raw_info = read_json(raw_marker)
            if decision_day is None:
                atomic_json(baseline_index, {"key": key, "signatures": raw_info["signatures"]})
                audit = {"role": "baseline", "pre_intervention_match": True, "checked_days": 0}
            else:
                if not baseline_index.is_file():
                    raise RuntimeError("Baseline must be prepared before its counterfactual arms")
                reference = read_json(baseline_index)
                mismatches = [d for d in range(1, decision_day + 1)
                              if raw_info["signatures"][str(d)] != reference["signatures"].get(str(d))]
                if mismatches:
                    raise RuntimeError(f"Causal prefix differs before intervention (seed={seed}, days={mismatches}); refusing to train")
                audit = {"role": "counterfactual", "pre_intervention_match": True,
                         "baseline_key": reference["key"], "checked_days": decision_day}
            atomic_json(arm / "branch_audit.json", audit)
            for mode in tracks:
                dest = arm / f"TRACK_{mode}"
                if validate_marker(dest) and set(indexed(dest, ".pt")) >= wanted:
                    continue
                for path in raw.glob("*.pt"):
                    path.unlink()
                command = _build_convert_cmd(python_bin=python_bin, graphml_dir=raw, horizons=horizons, state_mode=mode)
                command += ["--keep_graphml", "--workers", str(context["conversion_workers"])]
                execute(command, f"convert_{mode}.log")
                converted = indexed(raw, ".pt")
                if set(converted) != set(range(1, days + 1)):
                    raise RuntimeError(f"Incomplete PT conversion: {raw} ({mode})")
                dest.mkdir(exist_ok=True)
                (dest / "complete.json").unlink(missing_ok=True)
                for path in dest.glob("*.pt"):
                    path.unlink()
                for day in wanted:
                    copy_file(converted[day], dest / converted[day].name)
                atomic_json(dest / "complete.json", {"identity_sha256": key, "state_mode": mode,
                            "days": sorted(wanted), "files": describe_files(dest), "branch_audit": audit})
            for path in raw.glob("*.pt"):
                path.unlink()
            if not context["keep_raw_graphml"]:
                shutil.rmtree(raw)
        elif decision_day is None:
            raw_info = read_json(arm / "raw_complete.json")
            atomic_json(baseline_index, {"key": key, "signatures": raw_info["signatures"]})
        install_mode(arm / f"TRACK_{state_mode}", Path(out_dir), "arm_complete.json")
