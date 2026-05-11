#!/usr/bin/env python3
import argparse
import os
import time
import re
import csv
import json
import shutil
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Union, List, Dict, Any, Optional, Set

_GRAPH_FILENAME_RE_CACHE: Dict[str, re.Pattern[str]] = {}
_WORKER_Y_LOOKUP: Dict[str, Dict[str, float]] = {}
_WORKER_STATE_MODE: str = "ground_truth"
_WORKER_KEEP_GRAPHML: bool = False
_WORKER_PT_OUT_DIR: Optional[str] = None
_WORKER_RUN_TAG: str = ""
_WORKER_GRAPH_META: Dict[str, Dict[str, Any]] = {}
_WORKER_CF_DELTA_LOOKUP: Dict[str, Dict[str, float]] = {}

import math
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


# ============================================================
# Helpers: node-state counts and resistant fraction
# ============================================================
def _count_states(G: nx.Graph) -> Dict[str, int]:
    """
    Count node AMR states based on node attribute 'amr_state':
      0: U, 1: CS, 2: CR, 3: IS, 4: IR
    Returns counts for CS, CR, IS, IR (U is not used for denominator).
    """
    cs = cr = is_ = ir = 0
    for _, attrs in G.nodes(data=True):
        st = attrs.get("amr_state", 0)
        try:
            st_int = int(st)
        except Exception:
            st_int = 0
        if st_int == 1:
            cs += 1
        elif st_int == 2:
            cr += 1
        elif st_int == 3:
            is_ += 1
        elif st_int == 4:
            ir += 1
    return {"cs": cs, "cr": cr, "is": is_, "ir": ir}


def _resistant_fraction_from_counts(c: Dict[str, int]) -> float:
    denom = float(c["cs"] + c["cr"] + c["is"] + c["ir"])
    if denom <= 0.0:
        return 0.0
    return float(c["cr"] + c["ir"]) / denom


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_task_csv(path: str, rows: List[Tuple[str, Any]]) -> None:
    """
    Write a 2-column CSV with header: graphml,label
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["graphml", "label"])
        for graphml_name, label in rows:
            w.writerow([graphml_name, label])


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _choose_threshold_to_avoid_single_class(
    values: List[float],
    default_threshold: float
) -> Tuple[float, bool]:
    """
    Try to choose a threshold that yields both classes (0/1) if possible.

    Returns:
      (threshold_used, succeeded_avoiding_single_class)
    """
    if len(values) == 0:
        return default_threshold, False

    vals = np.asarray(values, dtype=float)
    candidates = [
        float(default_threshold),
        float(np.quantile(vals, 0.50)),  # median fallback
        float(np.quantile(vals, 0.25)),
        float(np.quantile(vals, 0.75)),
    ]

    for thr in candidates:
        y = (vals >= thr).astype(int)
        if int(y.min()) != int(y.max()):
            return float(thr), True

    return float(default_threshold), False


def _parse_horizons(horizons_str: str) -> List[int]:
    """
    Parse horizons from a comma-separated string like "7,14,21".
    Returns sorted unique positive ints.
    """
    if horizons_str is None:
        return []
    parts = [p.strip() for p in str(horizons_str).split(",") if p.strip() != ""]
    hs: Set[int] = set()
    for p in parts:
        if not re.fullmatch(r"\d+", p):
            raise ValueError(f"Invalid horizon token '{p}'. Expected integers like '7,14,21'.")
        h = int(p)
        if h <= 0:
            raise ValueError(f"Invalid horizon '{h}'. Horizons must be positive integers.")
        hs.add(h)
    return sorted(hs)


def _parse_graph_filename(fname: str, ext: str) -> Optional[Tuple[str, int, Optional[int]]]:
    """
    Parse filenames of the form:
        <sim_prefix>_t<day>[ _L<label> ]<ext>

    Returns (sim_prefix, day, label) or None if not matching.
    """
    e = str(ext)
    if e not in _GRAPH_FILENAME_RE_CACHE:
        pat = rf"^(?P<prefix>.+?)_t(?P<t>\d+)(?:_L(?P<label>\d+))?{re.escape(e)}$"
        _GRAPH_FILENAME_RE_CACHE[e] = re.compile(pat)
    m = _GRAPH_FILENAME_RE_CACHE[e].match(str(fname))
    if not m:
        return None
    prefix = str(m.group("prefix"))
    t = int(m.group("t"))
    lab = m.group("label")
    label = int(lab) if lab is not None else None
    return prefix, t, label


def _make_run_tag(graphml_dir: str) -> str:
    """
    Deterministic run tag derived from the absolute graphml_dir path.
    This makes trajectory prefixes globally unique across different runs/steps,
    without needing changes to generate_amr_data.py.
    """
    abs_dir = os.path.abspath(graphml_dir)
    base = os.path.basename(os.path.normpath(abs_dir))
    h = hashlib.sha1(abs_dir.encode("utf-8")).hexdigest()[:10]
    # Keep it filename-safe
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-")
    return f"{base}__{h}"


def _infer_region_from_prefix(prefix: str) -> Optional[int]:
    """
    If prefix contains 'r<digits>' (e.g., amr_r0), extract that.
    """
    m = re.search(r"(?:^|[_-])r(\d+)(?:$|[_-])", str(prefix))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_json_loads(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    s = str(value or '').strip()
    if s == '':
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _encode_multiplier_feature(value: Any, default: float = 1.0) -> float:
    """Neutral-centred encoding for strictly positive multiplier-like quantities."""
    z = _safe_float(value, default)
    if not np.isfinite(z) or z <= 0.0:
        return 0.0
    return float(math.log(z))


def _canonical_action_payload(meta: Dict[str, Any], graph_attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    graph_attrs = graph_attrs or {}
    raw_json = str(meta.get('cf_intervention_json', graph_attrs.get('cf_intervention_json', ''))).strip()
    payload = _safe_json_loads(raw_json)

    name = str(payload.get('name', meta.get('cf_intervention_name', graph_attrs.get('cf_intervention_name', '')))).strip()
    target_type = str(payload.get('target_type', meta.get('cf_intervention_target_type', graph_attrs.get('cf_intervention_target_type', '')))).strip()
    target_id = str(payload.get('target_id', meta.get('cf_intervention_target_id', graph_attrs.get('cf_intervention_target_id', '')))).strip()
    params = payload.get('params', {})
    params = dict(params) if isinstance(params, dict) else {}

    start_day = payload.get('start_day', meta.get('cf_intervention_start_day', graph_attrs.get('cf_intervention_start_day', '')) )
    end_day = payload.get('end_day', meta.get('cf_intervention_end_day', graph_attrs.get('cf_intervention_end_day', '')) )
    description = str(payload.get('description', meta.get('cf_intervention_description', graph_attrs.get('cf_intervention_description', '')))).strip()

    return {
        'name': name,
        'target_type': target_type,
        'target_id': target_id,
        'params': params,
        'start_day': start_day,
        'end_day': end_day,
        'description': description,
        'json': raw_json,
    }


def _build_action_features(payload: Dict[str, Any]) -> Tuple[List[float], str]:
    name = str(payload.get('name', '')).strip()
    target_type = str(payload.get('target_type', '')).strip().lower()
    target_id = str(payload.get('target_id', '')).strip()
    params = payload.get('params', {})
    params = dict(params) if isinstance(params, dict) else {}

    names = [
        'baseline_or_none',
        'reduce_ward_importation',
        'remove_staff_crossward_cover',
        'remove_specific_staff',
        'remove_edge',
        'set_screening_frequency',
        'set_screening_delay',
        'disable_isolation_response',
        'set_isolation_parameters',
    ]
    name_index = {n: i for i, n in enumerate(names)}
    onehot = [0.0] * len(names)
    if name == '' or name not in name_index:
        onehot[0] = 1.0
    else:
        onehot[name_index[name]] = 1.0

    is_global = 1.0 if target_type == 'global' else 0.0
    is_ward = 1.0 if target_type == 'ward' else 0.0
    is_staff = 1.0 if target_type == 'staff' else 0.0
    is_edge = 1.0 if target_type == 'edge' else 0.0
    is_policy = 1.0 if target_type == 'policy' else 0.0
    is_hospital = 1.0 if target_type == 'hospital' else 0.0

    freq_days = _safe_float(params.get('frequency_days', params.get('screen_every_k_days', params.get('k_days', 0.0))), 0.0)
    delay_days = _safe_float(params.get('delay_days', params.get('screen_result_delay_days', 0.0)), 0.0)
    multiplier = _safe_float(params.get('multiplier', params.get('multiplier_cr', 1.0)), 1.0)
    multiplier_cs = _safe_float(params.get('multiplier_cs', multiplier), multiplier)
    isolation_mult = _safe_float(params.get('isolation_mult', params.get('transmission_multiplier', multiplier)), multiplier)
    isolation_days = _safe_float(params.get('isolation_days', 0.0), 0.0)
    screen_on_admission = _safe_float(params.get('screen_on_admission', 0.0), 0.0)
    start_day = _safe_float(payload.get('start_day', 0.0), 0.0)
    end_day = _safe_float(payload.get('end_day', 0.0), 0.0)

    target_hash = int(hashlib.sha1(f'{target_type}|{target_id}'.encode('utf-8')).hexdigest()[:8], 16) if (target_type or target_id) else 0
    target_hash_norm = float(target_hash % 100000) / 100000.0

    feat = list(onehot) + [
        float(is_global),
        float(is_ward),
        float(is_staff),
        float(is_edge),
        float(is_policy),
        float(is_hospital),
        float(freq_days / 30.0),
        float(delay_days / 30.0),
        float(_encode_multiplier_feature(multiplier, 1.0)),
        float(_encode_multiplier_feature(multiplier_cs, 1.0)),
        float(_encode_multiplier_feature(isolation_mult, 1.0)),
        float(isolation_days / 30.0),
        float(max(0.0, min(1.0, screen_on_admission))),
        float(start_day / 365.0 if start_day > 0 else 0.0),
        float(end_day / 365.0 if end_day > 0 else 0.0),
        float(target_hash_norm),
    ]

    action_name = name if name != '' else 'baseline_or_none'
    if action_name == 'set_screening_frequency' and freq_days > 0:
        action_name = f'{action_name}__q{int(round(freq_days))}d'
    elif action_name == 'set_screening_delay' and delay_days >= 0:
        action_name = f'{action_name}__d{int(round(delay_days))}'
    elif action_name == 'set_isolation_parameters':
        action_name = (
            f"{action_name}__m{str(round(isolation_mult, 4)).replace('.', 'p')}"
            f"__d{int(round(isolation_days))}"
        )
    elif action_name == 'reduce_ward_importation' and target_id != '':
        action_name = f'{action_name}__{target_type}_{target_id}'

    return feat, action_name


def _cap_norm(value: Any, cap: float, default: float = 0.0) -> float:
    cap_f = float(cap) if float(cap) > 0.0 else 1.0
    raw = _safe_float(value, default)
    raw = max(0.0, min(raw, cap_f))
    return float(raw / cap_f)


def _build_operational_context_features(graph_attrs: Dict[str, Any]) -> torch.Tensor:
    attrs = dict(graph_attrs or {})
    screen_every_k_days = _cap_norm(attrs.get("current_screen_every_k_days", 0), 30.0, 0.0)
    weekly_screen_day = _cap_norm(attrs.get("current_weekly_screen_day", 0), 7.0, 0.0)
    screen_on_admission = _cap_norm(attrs.get("current_screen_on_admission", 0), 1.0, 0.0)
    screen_result_delay_days = _cap_norm(attrs.get("current_screen_result_delay_days", 0), 30.0, 0.0)
    isolation_mult = _encode_multiplier_feature(attrs.get("current_isolation_mult", 1.0), 1.0)
    isolation_days = _cap_norm(attrs.get("current_isolation_days", 0), 30.0, 0.0)
    persist_observations = _cap_norm(attrs.get("current_persist_observations", 0), 1.0, 0.0)
    is_screening_day = _cap_norm(attrs.get("current_is_screening_day", 0), 1.0, 0.0)
    days_until_next_screen = _cap_norm(attrs.get("current_days_until_next_screen", 0), 30.0, 0.0)

    return torch.tensor([
        screen_every_k_days,
        weekly_screen_day,
        screen_on_admission,
        screen_result_delay_days,
        isolation_mult,
        isolation_days,
        persist_observations,
        is_screening_day,
        days_until_next_screen,
    ], dtype=torch.float32)


def _read_graph_metadata(graphml_path: str) -> Dict[str, Any]:
    """
    Read graph-level metadata needed for causal pairing and robust trajectory bookkeeping.
    Uses a normal GraphML read for correctness; conversion already performs full reads later.
    """
    G = nx.read_graphml(graphml_path)
    fname = os.path.basename(graphml_path)
    parsed = _parse_graph_filename(fname, ".graphml")
    if parsed is None:
        raise ValueError(f"filename does not match '*_t<day>[ _L<label> ].graphml': {fname}")

    prefix, t_from_name, _ = parsed
    day = _safe_int(G.graph.get("day", t_from_name), t_from_name)

    region_attr = G.graph.get("region", None)
    if region_attr is not None:
        try:
            region = int(region_attr)
        except Exception:
            region = _infer_region_from_prefix(prefix)
    else:
        region = _infer_region_from_prefix(prefix)

    pair_id = str(G.graph.get("cf_pair_id", "")).strip()
    cf_role = str(G.graph.get("cf_role", "")).strip().lower()
    if cf_role not in {"factual", "counterfactual"}:
        cf_role = ""

    return {
        "filename": fname,
        "prefix": str(prefix),
        "day": int(day),
        "region": region,
        "cf_pair_id": pair_id,
        "cf_role": cf_role,
        "cf_shared_noise_seed": _safe_int(G.graph.get("cf_shared_noise_seed", 0), 0),
        "cf_intervention_name": str(G.graph.get("cf_intervention_name", "")).strip(),
        "cf_intervention_target_type": str(G.graph.get("cf_intervention_target_type", "")).strip(),
        "cf_intervention_target_id": str(G.graph.get("cf_intervention_target_id", "")).strip(),
        "cf_intervention_json": str(G.graph.get("cf_intervention_json", "")).strip(),
        "cf_intervention_start_day": G.graph.get("cf_intervention_start_day", ""),
        "cf_intervention_end_day": G.graph.get("cf_intervention_end_day", ""),
        "cf_intervention_description": str(G.graph.get("cf_intervention_description", "")).strip(),
        "cf_has_pair": 1 if pair_id != "" and cf_role in {"factual", "counterfactual"} else 0,
        "source_graphml": fname,
    }


def _build_counterfactual_delta_lookup(
    y_lookup: Dict[str, Dict[str, float]],
    graph_meta_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Build pair-level factual/counterfactual labels and deltas keyed by filename.

    Delta is always defined as: counterfactual - factual
    regardless of whether the current graph is the factual or counterfactual member.
    """
    out: Dict[str, Dict[str, float]] = {}
    groups: Dict[Tuple[str, int, Optional[int]], Dict[str, str]] = {}

    for fname, meta in graph_meta_lookup.items():
        pair_id = str(meta.get("cf_pair_id", "")).strip()
        role = str(meta.get("cf_role", "")).strip().lower()
        day = _safe_int(meta.get("day", 0), 0)
        region = meta.get("region", None)
        if pair_id == "" or role not in {"factual", "counterfactual"}:
            continue
        key = (pair_id, int(day), region)
        groups.setdefault(key, {})[role] = fname

    for _, role_map in groups.items():
        factual_fname = role_map.get("factual")
        counter_fname = role_map.get("counterfactual")
        if factual_fname is None or counter_fname is None:
            continue

        factual_labels = y_lookup.get(factual_fname, {})
        counter_labels = y_lookup.get(counter_fname, {})
        all_keys = sorted(set(factual_labels.keys()) | set(counter_labels.keys()))

        factual_bundle: Dict[str, float] = {"cf_has_complete_pair": 1.0}
        counter_bundle: Dict[str, float] = {"cf_has_complete_pair": 1.0}

        for key_name in all_keys:
            m = re.match(r"^h(\d+)_(.+)$", str(key_name))
            if not m:
                continue
            suffix = m.group(2)
            f_val = _safe_float(factual_labels.get(key_name, 0.0), 0.0)
            c_val = _safe_float(counter_labels.get(key_name, 0.0), 0.0)
            delta = c_val - f_val

            factual_bundle[f"{key_name}_factual"] = f_val
            factual_bundle[f"{key_name}_counterfactual"] = c_val
            factual_bundle[f"{key_name}_delta"] = delta

            counter_bundle[f"{key_name}_factual"] = f_val
            counter_bundle[f"{key_name}_counterfactual"] = c_val
            counter_bundle[f"{key_name}_delta"] = delta

            # Convenience aliases without duplicated horizon token when attaching later
            factual_bundle[f"cf_{key_name}_factual"] = f_val
            factual_bundle[f"cf_{key_name}_counterfactual"] = c_val
            factual_bundle[f"cf_{key_name}_delta"] = delta
            counter_bundle[f"cf_{key_name}_factual"] = f_val
            counter_bundle[f"cf_{key_name}_counterfactual"] = c_val
            counter_bundle[f"cf_{key_name}_delta"] = delta

        out[factual_fname] = factual_bundle
        out[counter_fname] = counter_bundle

    return out


# ============================================================
# Single-file conversion (LABEL PASSED EXPLICITLY)
# ============================================================


def _worker_init(
    y_lookup: Dict[str, Dict[str, float]],
    graph_meta_lookup: Dict[str, Dict[str, Any]],
    cf_delta_lookup: Dict[str, Dict[str, float]],
    state_mode: str,
    keep_graphml: bool,
    pt_out_dir: Optional[str],
    run_tag: str,
) -> None:
    global _WORKER_Y_LOOKUP, _WORKER_GRAPH_META, _WORKER_CF_DELTA_LOOKUP, _WORKER_STATE_MODE, _WORKER_KEEP_GRAPHML, _WORKER_PT_OUT_DIR, _WORKER_RUN_TAG
    _WORKER_Y_LOOKUP = y_lookup
    _WORKER_GRAPH_META = graph_meta_lookup
    _WORKER_CF_DELTA_LOOKUP = cf_delta_lookup
    _WORKER_STATE_MODE = state_mode
    _WORKER_KEEP_GRAPHML = keep_graphml
    _WORKER_PT_OUT_DIR = pt_out_dir
    _WORKER_RUN_TAG = run_tag


def _convert_one_worker(graphml_path: str) -> Union[Tuple[int, int, float], str]:
    return convert_one(
        graphml_path,
        _WORKER_Y_LOOKUP,
        _WORKER_GRAPH_META,
        _WORKER_CF_DELTA_LOOKUP,
        _WORKER_STATE_MODE,
        _WORKER_KEEP_GRAPHML,
        _WORKER_PT_OUT_DIR,
        _WORKER_RUN_TAG,
    )
def convert_one(
    graphml_path: str,
    y_lookup: Dict[str, Dict[str, float]],
    graph_meta_lookup: Dict[str, Dict[str, Any]],
    cf_delta_lookup: Dict[str, Dict[str, float]],
    state_mode: str,
    keep_graphml: bool,
    pt_out_dir: Optional[str],
    run_tag: str,
) -> Union[Tuple[int, int, float], str]:
    try:
        G = nx.read_graphml(graphml_path)

        fname = os.path.basename(graphml_path)
        parsed = _parse_graph_filename(fname, ".graphml")
        if parsed is None:
            return f"ERROR {graphml_path}: filename does not match '*_t<day>[ _L<label> ].graphml'"

        prefix, t_from_name, _ = parsed

        # Prefer graph attributes written by generate_amr_data.py, fall back to filename
        try:
            day = int(G.graph.get("day", t_from_name))
        except Exception:
            day = int(t_from_name)

        region_attr = G.graph.get("region", None)
        region: Optional[int]
        if region_attr is not None:
            try:
                region = int(region_attr)
            except Exception:
                region = _infer_region_from_prefix(prefix)
        else:
            region = _infer_region_from_prefix(prefix)

        graph_meta = dict(graph_meta_lookup.get(fname, {}))
        cf_labels = dict(cf_delta_lookup.get(fname, {}))
        cf_pair_id = str(graph_meta.get("cf_pair_id", G.graph.get("cf_pair_id", ""))).strip()
        cf_role = str(graph_meta.get("cf_role", G.graph.get("cf_role", ""))).strip().lower()
        if cf_role not in {"factual", "counterfactual"}:
            cf_role = ""
        cf_shared_noise_seed = _safe_int(
            graph_meta.get("cf_shared_noise_seed", G.graph.get("cf_shared_noise_seed", 0)),
            0,
        )
        cf_intervention_name = str(graph_meta.get("cf_intervention_name", G.graph.get("cf_intervention_name", ""))).strip()
        cf_intervention_target_type = str(graph_meta.get("cf_intervention_target_type", G.graph.get("cf_intervention_target_type", ""))).strip()
        cf_intervention_target_id = str(graph_meta.get("cf_intervention_target_id", G.graph.get("cf_intervention_target_id", ""))).strip()
        cf_has_pair = 1 if _safe_int(graph_meta.get("cf_has_pair", 0), 0) == 1 else 0
        cf_has_complete_pair = 1 if _safe_float(cf_labels.get("cf_has_complete_pair", 0.0), 0.0) > 0.0 else 0

        action_payload = _canonical_action_payload(graph_meta, graph_attrs=G.graph)
        action_features_list, action_name_compact = _build_action_features(action_payload)
        action_features = torch.tensor([action_features_list], dtype=torch.float32)

        # IMPORTANT: keep node order exactly as used for x and edge_index
        nodes = list(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}

        node_names: List[str] = [str(n) for n in nodes]
        node_roles: List[str] = []
        node_ward_id: List[int] = []
        node_ward_ids: List[str] = []
        node_ward_cover_count: List[int] = []
        for n in nodes:
            attrs_n = G.nodes[n]
            node_roles.append(str(attrs_n.get("role", "patient")))
            try:
                w_home = int(float(attrs_n.get("ward_id", 0)))
            except Exception:
                w_home = 0
            ward_ids_raw = str(attrs_n.get("ward_ids", str(w_home))).strip()
            if ward_ids_raw == "":
                ward_ids_raw = str(w_home)
            ward_tokens = [tok.strip() for tok in ward_ids_raw.split(",") if tok.strip() != ""]
            if len(ward_tokens) == 0:
                ward_tokens = [str(w_home)]
            node_ward_id.append(int(w_home))
            node_ward_ids.append(",".join(ward_tokens))
            node_ward_cover_count.append(int(len(ward_tokens)))

        N = len(nodes)

        # ---------------- Node features ----------------

        x_rows: List[List[float]] = []

        max_ward_id = max(node_ward_id) if len(node_ward_id) > 0 else 0
        max_ward_id = max(1, int(max_ward_id))
        max_cover = max(node_ward_cover_count) if len(node_ward_cover_count) > 0 else 0
        max_cover = max(1, int(max_cover))

        for idx, n in enumerate(nodes):
            attrs = G.nodes[n]
            role = attrs.get("role", "patient")
            is_staff = 1.0 if role == "staff" else 0.0

            amr_state = attrs.get("amr_state", 0)
            abx_class = attrs.get("abx_class", 0)
            is_isolated = attrs.get("is_isolated", 0)

            try:
                abx_f = float(abx_class)
            except Exception:
                abx_f = 0.0

            try:
                iso_f = float(is_isolated)
            except Exception:
                iso_f = 0.0

            try:
                new_cr_i = float(attrs.get("new_cr_acq_today", 0))
            except Exception:
                new_cr_i = 0.0

            try:
                new_ir_i = float(attrs.get("new_ir_inf_today", 0))
            except Exception:
                new_ir_i = 0.0

            obs_status = attrs.get("obs_status", 0)
            try:
                obs_i = int(float(obs_status))
            except Exception:
                obs_i = 0
            obs_positive = 1.0 if obs_i == 2 else 0.0
            obs_known = 1.0 if obs_i in (1, 2) else 0.0

            screened_today = 1.0 if _safe_int(attrs.get("screened_today", 0), 0) == 1 else 0.0
            days_since_last_test_norm = _cap_norm(attrs.get("days_since_last_test", 999), 30.0, 30.0)
            pending_test_days_norm = _cap_norm(attrs.get("pending_test_days", 0), 30.0, 0.0)
            pending_test_result = 1.0 if _safe_int(attrs.get("pending_test_result", 0), 0) == 1 else 0.0
            needs_admission_screen = 1.0 if _safe_int(attrs.get("needs_admission_screen", 0), 0) == 1 else 0.0
            present_today = 1.0 if _safe_int(attrs.get("present_today", 1), 1) == 1 else 0.0
            isolation_days_remaining_norm = _cap_norm(attrs.get("isolation_days_remaining", 0), 30.0, 0.0)
            admission_day_raw = attrs.get("admission_day", None)
            if admission_day_raw in (None, "", "null"):
                admission_age_norm = 1.0
            else:
                admission_age = max(0, int(day) - _safe_int(admission_day_raw, int(day)))
                admission_age_norm = _cap_norm(admission_age, 30.0, 30.0)

            ward_id_norm = float(node_ward_id[idx]) / float(max_ward_id)
            ward_cover_norm = float(node_ward_cover_count[idx]) / float(max_cover)

            if state_mode == "partial_observation":
                x_rows.append([
                    is_staff,
                    obs_positive,
                    abx_f,
                    iso_f,
                    0.0,
                    0.0,
                    ward_id_norm,
                    ward_cover_norm,
                    obs_known,
                    screened_today,
                    days_since_last_test_norm,
                    pending_test_days_norm,
                    needs_admission_screen,
                    present_today,
                    isolation_days_remaining_norm,
                    admission_age_norm,
                ])
            else:
                try:
                    amr_i = int(float(amr_state))
                except Exception:
                    amr_i = 0

                amr_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
                if 0 <= amr_i < 5:
                    amr_onehot[amr_i] = 1.0

                x_rows.append([
                    is_staff,
                    amr_onehot[0],
                    amr_onehot[1],
                    amr_onehot[2],
                    amr_onehot[3],
                    amr_onehot[4],
                    abx_f,
                    iso_f,
                    new_cr_i,
                    new_ir_i,
                    ward_id_norm,
                    ward_cover_norm,
                    obs_positive,
                    obs_known,
                    screened_today,
                    days_since_last_test_norm,
                    pending_test_days_norm,
                    pending_test_result,
                    needs_admission_screen,
                    present_today,
                    isolation_days_remaining_norm,
                    admission_age_norm,
                ])

        if x_rows:
            x = torch.tensor(x_rows, dtype=torch.float32)
        else:
            feat_dim = 16 if state_mode == "partial_observation" else 22
            x = torch.empty((0, feat_dim), dtype=torch.float32)

        # ---------------- Edges ----------------
        edges: List[List[int]] = []
        edge_attr: List[List[float]] = []

        for u, v, attrs in G.edges(data=True):
            ui = node_map[u]
            vi = node_map[v]
            edges.append([ui, vi])

            w = attrs.get("weight", 1.0)
            et = attrs.get("edge_type", 0)

            try:
                w_f = float(w)
            except Exception:
                w_f = 1.0
            try:
                et_f = float(et)
            except Exception:
                et_f = 0.0

            edge_attr.append([w_f, et_f])

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr_t = torch.empty((0, 2), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_t,
        )

        # Attach stable node identity info and ward-aware metadata used for post hoc attribution
        data.node_names = node_names
        data.node_roles = node_roles
        data.node_ward_id = torch.tensor(node_ward_id, dtype=torch.long)
        data.node_ward_cover_count = torch.tensor(node_ward_cover_count, dtype=torch.long)
        data.node_ward_ids = node_ward_ids

        # ------------------------------------------------------------
        # NEW: attach generation-grounded trajectory identity metadata
        # ------------------------------------------------------------
        # sim_id is the unique trajectory identifier used for temporal grouping
        # (one run folder + one region)
        if region is None:
            sim_id = f"{run_tag}__{prefix}"
        else:
            sim_id = f"{run_tag}__{prefix}__r{int(region)}"

        data.sim_id = str(sim_id)
        data.day = int(day)
        if region is not None:
            data.region = int(region)

        data.source_graphml = str(fname)
        data.graphml_dir_tag = str(run_tag)

        # ---------------- Causal metadata ----------------
        data.cf_pair_id = str(cf_pair_id)
        data.cf_role = str(cf_role)
        data.cf_has_pair = torch.tensor([int(cf_has_pair)], dtype=torch.long)
        data.cf_has_complete_pair = torch.tensor([int(cf_has_complete_pair)], dtype=torch.long)
        data.cf_is_factual = torch.tensor([1 if cf_role == "factual" else 0], dtype=torch.long)
        data.cf_is_counterfactual = torch.tensor([1 if cf_role == "counterfactual" else 0], dtype=torch.long)
        data.cf_shared_noise_seed = torch.tensor([int(cf_shared_noise_seed)], dtype=torch.long)
        data.cf_intervention_name = str(cf_intervention_name)
        data.cf_intervention_target_type = str(cf_intervention_target_type)
        data.cf_intervention_target_id = str(cf_intervention_target_id)
        data.cf_intervention_json = str(action_payload.get("json", ""))
        data.cf_intervention_start_day = torch.tensor([_safe_int(action_payload.get("start_day", 0), 0)], dtype=torch.long)
        data.cf_intervention_end_day = torch.tensor([_safe_int(action_payload.get("end_day", 0), 0)], dtype=torch.long)
        data.action_name = str(action_name_compact)
        data.action_target_type = str(action_payload.get("target_type", ""))
        data.action_target_id = str(action_payload.get("target_id", ""))
        data.action_features = action_features
        data.action_feature_encoding_version = 2
        data.operational_context_features = _build_operational_context_features(G.graph).view(1, -1)
        data.operational_context_encoding_version = 2

        # ---------------- Labels ----------------
        labels = y_lookup.get(fname, {})

        # Existing task labels (preserve exact behavior)
        data.y_h7_cr_acq = torch.tensor([labels.get("h7_cr_acq", 0.0)], dtype=torch.float32)
        data.y_h14_cr_acq = torch.tensor([labels.get("h14_cr_acq", 0.0)], dtype=torch.float32)
        data.y_h7_ir_inf = torch.tensor([labels.get("h7_ir_inf", 0.0)], dtype=torch.float32)

        # Outbreak (kept as regression signal; classification threshold lives in tasks.py)
        data.y_h7_outbreak_cr = torch.tensor([labels.get("h7_cr_acq", 0.0)], dtype=torch.float32)

        # New task labels (preserve exact behavior)
        any_res = 1.0 if (labels.get("h7_cr_acq", 0.0) + labels.get("h7_ir_inf", 0.0)) > 0.0 else 0.0
        data.y_h7_any_res_emergence = torch.tensor([any_res], dtype=torch.long)

        data.y_h7_total_inf = torch.tensor([labels.get("h7_ir_inf", 0.0)], dtype=torch.float32)

        # Resistant fraction within 14 days (continuous)
        data.y_h14_resistant_frac = torch.tensor([labels.get("h14_resistant_frac", 0.0)], dtype=torch.float32)

        # Resistant fraction within 14 days (binary, to avoid NaN ROC where possible)
        data.y_h14_resistant_frac_cls = torch.tensor([labels.get("h14_resistant_frac_cls", 0.0)], dtype=torch.long)

        # Antibiotic impact (placeholder but non-degenerate)
        data.y_h7_delta_res_if_abx_reduced = torch.tensor(
            [-0.1 * labels.get("h7_cr_acq", 0.0)], dtype=torch.float32
        )

        # Screening gain (placeholder; non-degenerate if h7_hidden_col varies)
        data.y_h7_screening_gain = torch.tensor(
            [1.0 if labels.get("h7_hidden_col", 0.0) > 0.0 else 0.0],
            dtype=torch.long
        )

        # Transmission attribution proxy
        data.y_true_transmissions = torch.tensor(
            [labels.get("h7_transmissions", 0.0)], dtype=torch.float32
        )

        # Dynamic census attribution (existing h7; preserve exact behavior)
        data.y_h7_trans_res = torch.tensor([labels.get("h7_trans_res", 0.0)], dtype=torch.float32)
        data.y_h7_import_res = torch.tensor([labels.get("h7_import_res", 0.0)], dtype=torch.float32)
        data.y_h7_trans_import_res = torch.tensor([
            labels.get("h7_trans_import_res", labels.get("h7_trans_res", 0.0) + labels.get("h7_import_res", 0.0))
        ], dtype=torch.float32)
        data.y_h7_select_res = torch.tensor([labels.get("h7_select_res", 0.0)], dtype=torch.float32)

        data.y_h7_trans_share = torch.tensor([labels.get("h7_trans_share", 0.0)], dtype=torch.float32)
        data.y_h7_endog_share = torch.tensor([labels.get("h7_endog_share", 0.0)], dtype=torch.float32)
        data.y_h7_import_share = torch.tensor([labels.get("h7_import_share", 0.0)], dtype=torch.float32)
        data.y_h7_select_share = torch.tensor([labels.get("h7_select_share", 0.0)], dtype=torch.float32)

        data.y_h7_trans_majority = torch.tensor([labels.get("h7_trans_majority", 0.0)], dtype=torch.long)
        data.y_h7_endog_majority = torch.tensor([labels.get("h7_endog_majority", 0.0)], dtype=torch.long)

        # ------------------------------------------------------------
        # NEW: attach additional horizonised labels dynamically
        # without overriding any existing attributes.
        # Also attach causal paired labels when available.
        # ------------------------------------------------------------
        long_suffixes = {
            "any_res_emergence",
            "resistant_frac_cls",
            "screening_gain",
            "trans_majority",
            "endog_majority",
        }

        for source_dict in (labels, cf_labels):
            for k, v in source_dict.items():
                m = re.match(r"^(?:cf_)?h(\d+)_(.+)$", str(k))
                if not m:
                    continue

                if str(k).startswith("cf_"):
                    attr = f"y_{str(k)}"
                    suffix = str(k)
                else:
                    h = m.group(1)
                    suffix = m.group(2)
                    attr = f"y_h{h}_{suffix}"

                if hasattr(data, attr):
                    continue

                if suffix.endswith("_factual") or suffix.endswith("_counterfactual") or suffix.endswith("_delta"):
                    data_val = torch.tensor([float(v)], dtype=torch.float32)
                elif suffix in long_suffixes:
                    data_val = torch.tensor([int(float(v))], dtype=torch.long)
                else:
                    data_val = torch.tensor([float(v)], dtype=torch.float32)

                setattr(data, attr, data_val)

        # Keep source graphml filename for backward compatibility
        data.filename = fname

        # ------------------------------------------------------------
        # Output naming: make prefixes globally unique across runs
        # ------------------------------------------------------------
        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", str(prefix)).strip("-")
        safe_run_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_tag)).strip("-")
        out_base = f"{safe_run_tag}__{safe_prefix}_t{int(day)}.pt"

        out_path = os.path.join(os.path.dirname(graphml_path), out_base)
        torch.save(data, out_path)

        # Optional: copy into a mode-specific archive directory to avoid overwrites across tracks
        if pt_out_dir is not None and str(pt_out_dir).strip() != "":
            _ensure_dir(pt_out_dir)
            archived_path = os.path.join(pt_out_dir, os.path.basename(out_path))
            try:
                shutil.copy2(out_path, archived_path)
            except Exception as e:
                return f"ERROR {graphml_path}: failed to copy PT to pt_out_dir '{pt_out_dir}': {e}"

        if not keep_graphml:
            os.remove(graphml_path)

        degrees = [d for _, d in G.degree()]
        return (N, len(edges), float(np.mean(degrees) if len(degrees) > 0 else 0.0))

    except Exception as e:
        return f"ERROR {graphml_path}: {e}"


# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--state_mode",
        type=str,
        choices=["ground_truth", "partial_observation"],
        default=os.environ.get("DT_STATE_MODE", "ground_truth"),
        help=(
            "Controls what node-state signal is fed into x[:,1]. "
            "ground_truth uses latent amr_state (0..4). "
            "partial_observation uses obs_status mapped to binary known-positive (1) else 0. "
            "You can also set DT_STATE_MODE in the environment."
        ),
    )
    parser.add_argument(
        "--keep_graphml",
        action="store_true",
        default=bool(int(os.environ.get("DT_KEEP_GRAPHML", "0"))),
        help=(
            "If set, do NOT delete .graphml after writing .pt. "
            "Default can also be controlled via DT_KEEP_GRAPHML=1."
        ),
    )
    parser.add_argument(
        "--pt_out_dir",
        type=str,
        default=os.environ.get("DT_PT_OUT_DIR", ""),
        help=(
            "Optional: directory to also copy each generated .pt into (mode-specific archive). "
            "Primary .pt is written into graphml_dir (not next to each .graphml filename). "
            "Default can also be controlled via DT_PT_OUT_DIR."
        ),
    )

    parser.add_argument(
        "--label_csv_dir",
        type=str,
        default=None,
        help="Directory to write per-task label CSVs (default: <graphml_dir>/labels).",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="7,14",
        help=(
            "Comma-separated prediction horizons (days) to compute labels for, e.g. '7,14,21,30'. "
            "Legacy horizons 7 and 14 are always included to preserve existing functionality."
        ),
    )
    parser.add_argument(
        "--early_res_frac_threshold",
        type=float,
        default=0.15,
        help="Default threshold for early_outbreak_warning_h14 binary label.",
    )
    parser.add_argument(
        "--early_res_frac_threshold_file",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing a persisted threshold for the early_outbreak_warning_h14 label. "
            "If provided (and exists), the stored threshold is used instead of recomputing from the current folder."
        ),
    )
    parser.add_argument(
        "--early_res_frac_threshold_out",
        type=str,
        default=None,
        help=(
            "Optional JSON output path to persist the chosen early_outbreak_warning_h14 threshold. "
            "Use the resulting file for consistent train/test semantics (pass via --early_res_frac_threshold_file)."
        ),
    )
    args = parser.parse_args()

    state_mode = str(getattr(args, "state_mode", "ground_truth")).strip()
    if state_mode not in {"ground_truth", "partial_observation"}:
        raise ValueError(f"Invalid --state_mode: {state_mode}")

    keep_graphml = bool(getattr(args, "keep_graphml", False))

    pt_out_dir = str(getattr(args, "pt_out_dir", "")).strip()
    if pt_out_dir == "":
        pt_out_dir_use: Optional[str] = None
    else:
        pt_out_dir_use = os.path.abspath(pt_out_dir)
        _ensure_dir(pt_out_dir_use)

    graphml_dir = os.path.abspath(args.graphml_dir)
    run_tag = _make_run_tag(graphml_dir)

    files = sorted(f for f in os.listdir(graphml_dir) if f.endswith(".graphml"))

    graph_meta_lookup: Dict[str, Dict[str, Any]] = {}
    for f in files:
        graphml_path = os.path.join(graphml_dir, f)
        try:
            graph_meta_lookup[f] = _read_graph_metadata(graphml_path)
        except Exception as exc:
            print(f"DT_CONV_WARN failed to read graph metadata for {f}: {exc}", flush=True)
            graph_meta_lookup[f] = {
                "filename": f,
                "prefix": "",
                "day": 0,
                "region": None,
                "cf_pair_id": "",
                "cf_role": "",
                "cf_shared_noise_seed": 0,
                "cf_intervention_name": "",
                "cf_intervention_target_type": "",
                "cf_intervention_target_id": "",
                "cf_has_pair": 0,
                "source_graphml": f,
            }


    label_csv_dir = args.label_csv_dir
    if label_csv_dir is None:
        label_csv_dir = os.path.join(graphml_dir, "labels")
    label_csv_dir = os.path.abspath(label_csv_dir)
    _ensure_dir(label_csv_dir)

    user_horizons = _parse_horizons(args.horizons)
    horizons_set: Set[int] = set(user_horizons)
    horizons_set.add(7)
    horizons_set.add(14)
    horizons = sorted(horizons_set)

    print(
        f"DT_CONV_META files={len(files)} horizons={','.join(str(h) for h in horizons)} "
        f"state_mode={state_mode} keep_graphml={int(keep_graphml)} "
        f"pt_out_dir={'none' if pt_out_dir_use is None else pt_out_dir_use} run_tag={run_tag}",
        flush=True,
    )

    _write_json(
        os.path.join(label_csv_dir, "manifest.json"),
        {
            "horizons": [int(h) for h in horizons],
            "n_graphml_files": int(len(files)),
            "created_unix": int(time.time()),
            "state_mode": str(state_mode),
            "keep_graphml": bool(keep_graphml),
            "pt_out_dir": "" if pt_out_dir_use is None else str(pt_out_dir_use),
            "run_tag": str(run_tag),
            "graphml_dir": str(graphml_dir),
        },
    )

    # ------------------------------------------------------------
    # FIRST PASS: compute horizon labels
    # ------------------------------------------------------------
    y_lookup: Dict[str, Dict[str, float]] = {}
    sim_groups: Dict[Tuple[str, Optional[int], str, str], List[tuple]] = {}

    for f in files:
        parsed = _parse_graph_filename(f, ".graphml")
        if parsed is None:
            continue
        prefix, t, _ = parsed
        meta = graph_meta_lookup.get(f, {})
        region = meta.get("region", _infer_region_from_prefix(prefix))
        cf_pair_id = str(meta.get("cf_pair_id", "")).strip()
        cf_role = str(meta.get("cf_role", "")).strip().lower()

        G = nx.read_graphml(os.path.join(graphml_dir, f))

        cr_evt = int(G.graph.get("new_cr_acq_total", 0))
        ir_evt = int(G.graph.get("new_ir_inf_total", 0))

        import_cr_evt = int(G.graph.get("new_import_cr_total", 0))
        import_ir_evt = int(G.graph.get("new_import_ir_total", 0))
        import_res_evt = int(import_cr_evt + import_ir_evt)
        trans_cr_evt = int(G.graph.get("new_trans_cr_total", 0))
        select_cr_evt = int(G.graph.get("new_select_cr_total", 0))

        counts = _count_states(G)
        res_frac = _resistant_fraction_from_counts(counts)

        group_key = (str(prefix), region, cf_pair_id, cf_role)
        sim_groups.setdefault(group_key, []).append(
            (t, f, cr_evt, ir_evt, res_frac, import_res_evt, trans_cr_evt, select_cr_evt)
        )

    for seq in sim_groups.values():
        seq.sort(key=lambda z: z[0])

        for i, (_, fname, _, _, _, _, _, _) in enumerate(seq):
            y_lookup.setdefault(fname, {})

            for H in horizons:
                t_i = int(seq[i][0])
                futureH = []
                for k in range(1, int(H) + 1):
                    t_needed = t_i + k
                    if i + k >= len(seq) or int(seq[i + k][0]) != t_needed:
                        break
                    futureH.append(seq[i + k])

                h_cr_acq = float(sum(cr for _, _, cr, _, _, _, _, _ in futureH))
                h_ir_inf = float(sum(ir for _, _, _, ir, _, _, _, _ in futureH))

                h_hidden_col = h_cr_acq
                h_transmissions = float(sum((cr + ir) for _, _, cr, ir, _, _, _, _ in futureH))

                h_import_res = float(sum(imp for _, _, _, _, _, imp, _, _ in futureH))
                h_trans_res = float(sum(tr for _, _, _, _, _, _, tr, _ in futureH))
                h_select_res = float(sum(sel for _, _, _, _, _, _, _, sel in futureH))
                h_endog_res = float(h_trans_res + h_select_res)

                denom_trans_import = float(h_trans_res + h_import_res)
                if denom_trans_import <= 0.0:
                    trans_share = 0.0
                else:
                    trans_share = float(h_trans_res) / denom_trans_import

                denom_endog_import = float(h_endog_res + h_import_res)
                if denom_endog_import <= 0.0:
                    endog_share = 0.0
                else:
                    endog_share = float(h_endog_res) / denom_endog_import

                denom_3way = float(h_import_res + h_trans_res + h_select_res)
                if denom_3way <= 0.0:
                    import_share_3 = 0.0
                    trans_share_3 = 0.0
                    select_share_3 = 0.0
                else:
                    import_share_3 = float(h_import_res) / denom_3way
                    trans_share_3 = float(h_trans_res) / denom_3way
                    select_share_3 = float(h_select_res) / denom_3way

                if len(futureH) > 0:
                    h_res_frac = float(max(rf for _, _, _, _, rf, _, _, _ in futureH))
                else:
                    h_res_frac = 0.0

                y_lookup[fname][f"h{H}_cr_acq"] = h_cr_acq
                y_lookup[fname][f"h{H}_ir_inf"] = h_ir_inf
                y_lookup[fname][f"h{H}_hidden_col"] = h_hidden_col
                y_lookup[fname][f"h{H}_transmissions"] = h_transmissions

                y_lookup[fname][f"h{H}_import_res"] = h_import_res
                y_lookup[fname][f"h{H}_trans_res"] = h_trans_res
                y_lookup[fname][f"h{H}_trans_import_res"] = float(h_trans_res + h_import_res)
                y_lookup[fname][f"h{H}_select_res"] = h_select_res
                y_lookup[fname][f"h{H}_endog_res"] = h_endog_res

                y_lookup[fname][f"h{H}_trans_share"] = float(trans_share)
                y_lookup[fname][f"h{H}_endog_share"] = float(endog_share)
                y_lookup[fname][f"h{H}_trans_majority"] = 1.0 if float(trans_share) >= 0.5 else 0.0
                y_lookup[fname][f"h{H}_endog_majority"] = 1.0 if float(endog_share) >= 0.5 else 0.0

                y_lookup[fname][f"h{H}_import_share"] = float(import_share_3)
                y_lookup[fname][f"h{H}_select_share"] = float(select_share_3)

                y_lookup[fname][f"h{H}_resistant_frac"] = h_res_frac

                y_lookup[fname][f"h{H}_any_res_emergence"] = (
                    1.0 if float(h_cr_acq + h_ir_inf) > 0.0 else 0.0
                )
                y_lookup[fname][f"h{H}_total_inf"] = float(h_ir_inf)
                y_lookup[fname][f"h{H}_delta_res_if_abx_reduced"] = float(-0.1 * h_cr_acq)
                y_lookup[fname][f"h{H}_screening_gain"] = (1.0 if float(h_hidden_col) > 0.0 else 0.0)

            y_lookup[fname]["h7_cr_acq"] = float(y_lookup[fname].get("h7_cr_acq", 0.0))
            y_lookup[fname]["h14_cr_acq"] = float(y_lookup[fname].get("h14_cr_acq", 0.0))
            y_lookup[fname]["h7_ir_inf"] = float(y_lookup[fname].get("h7_ir_inf", 0.0))
            y_lookup[fname]["h7_hidden_col"] = float(y_lookup[fname].get("h7_hidden_col", 0.0))
            y_lookup[fname]["h7_transmissions"] = float(y_lookup[fname].get("h7_transmissions", 0.0))
            y_lookup[fname]["h14_resistant_frac"] = float(y_lookup[fname].get("h14_resistant_frac", 0.0))

            y_lookup[fname]["h7_import_res"] = float(y_lookup[fname].get("h7_import_res", 0.0))
            y_lookup[fname]["h7_trans_res"] = float(y_lookup[fname].get("h7_trans_res", 0.0))
            y_lookup[fname]["h7_trans_import_res"] = float(
                y_lookup[fname].get("h7_trans_import_res", y_lookup[fname].get("h7_trans_res", 0.0) + y_lookup[fname].get("h7_import_res", 0.0))
            )
            y_lookup[fname]["h7_select_res"] = float(y_lookup[fname].get("h7_select_res", 0.0))
            y_lookup[fname]["h7_endog_res"] = float(y_lookup[fname].get("h7_endog_res", 0.0))

            y_lookup[fname]["h7_trans_share"] = float(y_lookup[fname].get("h7_trans_share", 0.0))
            y_lookup[fname]["h7_endog_share"] = float(y_lookup[fname].get("h7_endog_share", 0.0))
            y_lookup[fname]["h7_trans_majority"] = float(y_lookup[fname].get("h7_trans_majority", 0.0))
            y_lookup[fname]["h7_endog_majority"] = float(y_lookup[fname].get("h7_endog_majority", 0.0))

            y_lookup[fname]["h7_import_share"] = float(y_lookup[fname].get("h7_import_share", 0.0))
            y_lookup[fname]["h7_select_share"] = float(y_lookup[fname].get("h7_select_share", 0.0))

    # ------------------------------------------------------------
    # Derive binary early outbreak label (with optional persistence)
    # ------------------------------------------------------------
    all_h14_res_frac = [float(v.get("h14_resistant_frac", 0.0)) for v in y_lookup.values()]

    thr_used: Optional[float] = None
    thr_source = "computed"
    thr_path_used: Optional[str] = None

    if args.early_res_frac_threshold_file is not None:
        thr_file = os.path.abspath(args.early_res_frac_threshold_file)
        if os.path.isfile(thr_file):
            obj = _load_json(thr_file)
            if "h14_resistant_frac_threshold" in obj:
                thr_used = float(obj["h14_resistant_frac_threshold"])
            elif "threshold" in obj:
                thr_used = float(obj["threshold"])
            else:
                raise ValueError(
                    f"Threshold file '{thr_file}' is missing key 'h14_resistant_frac_threshold' (or 'threshold')."
                )
            thr_source = "loaded"
            thr_path_used = thr_file
        else:
            raise FileNotFoundError(f"Threshold file not found: '{thr_file}'")

    if thr_used is None:
        auto_candidate = os.path.join(label_csv_dir, "early_warning_threshold.json")
        if os.path.isfile(auto_candidate):
            obj = _load_json(auto_candidate)
            if "h14_resistant_frac_threshold" in obj:
                thr_used = float(obj["h14_resistant_frac_threshold"])
                thr_source = "auto_loaded"
                thr_path_used = auto_candidate

    if thr_used is None:
        thr_used, _ = _choose_threshold_to_avoid_single_class(
            values=all_h14_res_frac,
            default_threshold=float(args.early_res_frac_threshold),
        )
        thr_source = "computed"

    ok_balanced = False
    if len(all_h14_res_frac) > 0:
        vals = np.asarray(all_h14_res_frac, dtype=float)
        y_tmp = (vals >= float(thr_used)).astype(int)
        ok_balanced = bool(int(y_tmp.min()) != int(y_tmp.max()))

    msg = f"DT_CONV_EARLY_THR used={float(thr_used):.6f} balanced={int(ok_balanced)} source={thr_source}"
    if thr_path_used is not None:
        msg += f" file={thr_path_used}"
    print(msg, flush=True)

    for fname in list(y_lookup.keys()):
        y_lookup[fname]["h14_resistant_frac_cls"] = (
            1.0 if float(y_lookup[fname].get("h14_resistant_frac", 0.0)) >= float(thr_used) else 0.0
        )

    # OPTIONAL: Write per-label CSVs for auditability
    label_keys_to_export = []
    for H in horizons:
        label_keys_to_export.extend([
            f"h{H}_cr_acq",
            f"h{H}_ir_inf",
            f"h{H}_resistant_frac",
            f"h{H}_any_res_emergence",
            f"h{H}_total_inf",
            f"h{H}_delta_res_if_abx_reduced",
            f"h{H}_screening_gain",
            f"h{H}_transmissions",
            f"h{H}_import_res",
            f"h{H}_trans_res",
            f"h{H}_trans_import_res",
            f"h{H}_select_res",
            f"h{H}_endog_res",
            f"h{H}_trans_share",
            f"h{H}_endog_share",
            f"h{H}_import_share",
            f"h{H}_select_share",
            f"h{H}_trans_majority",
            f"h{H}_endog_majority",
        ])
    label_keys_to_export.append("h14_resistant_frac_cls")

    for key in label_keys_to_export:
        rows = [(fname, y_lookup.get(fname, {}).get(key, 0.0)) for fname in files]
        out_csv = os.path.join(label_csv_dir, f"{key}.csv")
        _write_task_csv(out_csv, rows)


    # ------------------------------------------------------------
    # Build paired factual/counterfactual labels when available
    # ------------------------------------------------------------
    cf_delta_lookup = _build_counterfactual_delta_lookup(y_lookup, graph_meta_lookup)
    paired_files = sum(
        1
        for v in cf_delta_lookup.values()
        if _safe_float(v.get("cf_has_complete_pair", 0.0), 0.0) > 0.0
    )
    print(f"DT_CONV_CAUSAL_PAIRS matched_files={paired_files}", flush=True)

    # Export causal metadata and paired labels for auditability when present
    causal_rows: List[Dict[str, Any]] = []
    causal_header = [
        "graphml",
        "cf_pair_id",
        "cf_role",
        "cf_has_pair",
        "cf_has_complete_pair",
        "cf_shared_noise_seed",
        "cf_intervention_name",
        "cf_intervention_target_type",
        "cf_intervention_target_id",
        "cf_intervention_json",
    ]
    for fname in files:
        meta = graph_meta_lookup.get(fname, {})
        paired = cf_delta_lookup.get(fname, {})
        row = {
            "graphml": fname,
            "cf_pair_id": str(meta.get("cf_pair_id", "")),
            "cf_role": str(meta.get("cf_role", "")),
            "cf_has_pair": int(_safe_int(meta.get("cf_has_pair", 0), 0)),
            "cf_has_complete_pair": int(_safe_float(paired.get("cf_has_complete_pair", 0.0), 0.0) > 0.0),
            "cf_shared_noise_seed": int(_safe_int(meta.get("cf_shared_noise_seed", 0), 0)),
            "cf_intervention_name": str(meta.get("cf_intervention_name", "")),
            "cf_intervention_target_type": str(meta.get("cf_intervention_target_type", "")),
            "cf_intervention_target_id": str(meta.get("cf_intervention_target_id", "")),
            "cf_intervention_json": str(meta.get("cf_intervention_json", "")),
        }
        causal_rows.append(row)

    with open(os.path.join(label_csv_dir, "causal_metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=causal_header)
        w.writeheader()
        for row in causal_rows:
            w.writerow(row)

    if cf_delta_lookup:
        paired_summary: List[Tuple[str, Any]] = []
        delta_export_keys = sorted({k for v in cf_delta_lookup.values() for k in v.keys() if k.endswith("_delta")})
        for key in delta_export_keys:
            paired_summary = [(fname, cf_delta_lookup.get(fname, {}).get(key, 0.0)) for fname in files]
            out_csv = os.path.join(label_csv_dir, f"{key}.csv")
            _write_task_csv(out_csv, paired_summary)

    if args.early_res_frac_threshold_out is not None:
        out_thr = os.path.abspath(args.early_res_frac_threshold_out)
        _write_json(out_thr, {"h14_resistant_frac_threshold": float(thr_used)})
    else:
        _write_json(
            os.path.join(label_csv_dir, "early_warning_threshold.json"),
            {"h14_resistant_frac_threshold": float(thr_used)}
        )


    # ------------------------------------------------------------
    # SECOND PASS: convert all graphml -> pt
    # ------------------------------------------------------------
    t0 = time.time()
    stats = {"ok": 0, "err": 0, "nodes": [], "edges": [], "deg": []}

    if args.workers and args.workers > 0:
        graphml_paths = [os.path.join(graphml_dir, f) for f in files]
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_worker_init,
            initargs=(y_lookup, graph_meta_lookup, cf_delta_lookup, state_mode, keep_graphml, pt_out_dir_use, run_tag),
        ) as ex:
            futs = [ex.submit(_convert_one_worker, graphml_path) for graphml_path in graphml_paths]
            for fut in as_completed(futs):
                res = fut.result()
                if isinstance(res, str) and res.startswith("ERROR"):
                    stats["err"] += 1
                    print(res, flush=True)
                else:
                    n_nodes, n_edges, mean_deg = res
                    stats["ok"] += 1
                    stats["nodes"].append(n_nodes)
                    stats["edges"].append(n_edges)
                    stats["deg"].append(mean_deg)
    else:
        for f in files:
            res = convert_one(
                os.path.join(graphml_dir, f),
                y_lookup,
                graph_meta_lookup,
                cf_delta_lookup,
                state_mode,
                keep_graphml,
                pt_out_dir_use,
                run_tag,
            )
            if isinstance(res, str) and res.startswith("ERROR"):
                stats["err"] += 1
                print(res, flush=True)
            else:
                n_nodes, n_edges, mean_deg = res
                stats["ok"] += 1
                stats["nodes"].append(n_nodes)
                stats["edges"].append(n_edges)
                stats["deg"].append(mean_deg)

    dt = time.time() - t0
    print(
        f"DT_CONV_DONE ok={stats['ok']} err={stats['err']} "
        f"mean_nodes={float(np.mean(stats['nodes'])) if stats['nodes'] else 0.0:.2f} "
        f"mean_edges={float(np.mean(stats['edges'])) if stats['edges'] else 0.0:.2f} "
        f"sec={dt:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
