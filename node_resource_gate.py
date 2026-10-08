#!/usr/bin/env python3
"""Canonical fail-closed CPU/RAM/GPU gate for run-lab mutations."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple


POLICY_VERSION = "run-lab-node-resource-gate-v1"
CPU_BLOCK_AT_PCT = 85.0
RAM_BLOCK_AT_PCT = 85.0


class ResourceObservationError(RuntimeError):
    """A required node-wide observation could not be established."""


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def _valid_pct(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 100.0
    )


def assess_node_resources(
    cpu_used_pct: object,
    ram_used_pct: object,
    *,
    observed_at: Optional[str] = None,
) -> Dict[str, object]:
    """Return pass/blocked/unknown using inclusive 85% CPU and RAM limits."""

    reasons = []
    if not _valid_pct(cpu_used_pct):
        reasons.append("cpu_usage_unavailable")
    if not _valid_pct(ram_used_pct):
        reasons.append("ram_usage_unavailable")
    if reasons:
        decision = "unknown"
    else:
        cpu = float(cpu_used_pct)
        ram = float(ram_used_pct)
        if cpu >= CPU_BLOCK_AT_PCT:
            reasons.append("cpu_at_or_above_85_pct")
        if ram >= RAM_BLOCK_AT_PCT:
            reasons.append("ram_at_or_above_85_pct")
        decision = "blocked" if reasons else "pass"

    return {
        "policy_version": POLICY_VERSION,
        "decision": decision,
        "reasons": reasons,
        "cpu_used_pct": (
            round(float(cpu_used_pct), 3) if _valid_pct(cpu_used_pct) else None
        ),
        "ram_used_pct": (
            round(float(ram_used_pct), 3) if _valid_pct(ram_used_pct) else None
        ),
        "thresholds": {
            "cpu_block_at_pct": CPU_BLOCK_AT_PCT,
            "ram_block_at_pct": RAM_BLOCK_AT_PCT,
            "comparison": ">=",
        },
        "observed_at": observed_at or now_iso(),
    }


def add_gpu_evidence(
    node_decision: Dict[str, object],
    gpu_snapshot: Optional[Iterable[Dict[str, object]]],
    *,
    selected_gpu: Optional[int],
    gpu_required: bool,
    query_error: Optional[str] = None,
) -> Dict[str, object]:
    """Add selected-GPU compute evidence without guessing when observation is incomplete.

    ``compute_processes`` comes from ``nvidia-smi --query-compute-apps``. Graphics-only clients
    that appear only in the GPU memory inventory are reported with the row but do not constitute
    compute occupancy.
    """

    result = dict(node_decision)
    rows = list(gpu_snapshot or [])
    result.update({
        "gpu_required": gpu_required,
        "selected_gpu": selected_gpu,
        "gpu_snapshot": rows,
        "container_attribution": "pending_inspect_lab",
    })
    reasons = list(result.get("reasons", []))
    if gpu_required:
        if query_error:
            reasons.append("gpu_usage_unavailable: {}".format(query_error))
            result["decision"] = "unknown"
        elif not rows or any(
            not isinstance(row, dict)
            or not isinstance(row.get("compute_processes"), list)
            or isinstance(row.get("index"), bool)
            or not str(row.get("index", "")).isdigit()
            for row in rows
        ):
            reasons.append("gpu_snapshot_incomplete")
            result["decision"] = "unknown"
        elif selected_gpu is None:
            reasons.append("selected_gpu_unresolved")
            result["decision"] = "unknown"
        else:
            selected = [row for row in rows if int(row["index"]) == selected_gpu]
            if len(selected) != 1:
                reasons.append("selected_gpu_not_observed")
                result["decision"] = "unknown"
            elif selected[0].get("compute_processes"):
                reasons.append("selected_gpu_has_compute_process")
                if result.get("decision") != "unknown":
                    result["decision"] = "blocked"
    result["reasons"] = reasons
    return result


def _read_cpu_times(path: Path = Path("/proc/stat")) -> Tuple[int, int]:
    try:
        fields = path.read_text(encoding="utf-8").splitlines()[0].split()
        if not fields or fields[0] != "cpu" or len(fields) < 5:
            raise ValueError("missing aggregate cpu row")
        values = [int(value) for value in fields[1:]]
    except (OSError, ValueError, IndexError) as error:
        raise ResourceObservationError("could not read aggregate CPU counters") from error
    # guest/guest_nice are already included in user/nice and must not be counted twice.
    return values[3], sum(values[:8])


def _read_ram_used_pct(path: Path = Path("/proc/meminfo")) -> float:
    values: Dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] in ("MemTotal:", "MemAvailable:"):
                values[fields[0]] = int(fields[1])
        total = values["MemTotal:"]
        available = values["MemAvailable:"]
        if total <= 0 or not 0 <= available <= total:
            raise ValueError("invalid memory totals")
    except (OSError, ValueError, KeyError) as error:
        raise ResourceObservationError("could not read MemTotal/MemAvailable") from error
    return 100.0 * (total - available) / total


def sample_cpu_ram(
    *,
    interval_seconds: float = 1.0,
    cpu_reader: Callable[[], Tuple[int, int]] = _read_cpu_times,
    ram_reader: Callable[[], float] = _read_ram_used_pct,
    sleeper: Callable[[float], None] = time.sleep,
) -> Dict[str, object]:
    """Measure one node-wide CPU delta and current RAM use; never return partial success."""

    if interval_seconds <= 0:
        raise ResourceObservationError("CPU sampling interval must be positive")
    idle_before, total_before = cpu_reader()
    sleeper(interval_seconds)
    idle_after, total_after = cpu_reader()
    delta_total = total_after - total_before
    delta_idle = idle_after - idle_before
    if delta_total <= 0 or delta_idle < 0 or delta_idle > delta_total:
        raise ResourceObservationError("invalid aggregate CPU counter delta")
    cpu_used_pct = 100.0 * (delta_total - delta_idle) / delta_total
    ram_used_pct = ram_reader()
    decision = assess_node_resources(cpu_used_pct, ram_used_pct)
    if decision["decision"] == "unknown":
        raise ResourceObservationError("invalid CPU/RAM percentage")
    return decision


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    assess = sub.add_parser("assess", help="assess already observed node-wide percentages")
    assess.add_argument("--cpu-used-pct", type=float)
    assess.add_argument("--ram-used-pct", type=float)
    assess.add_argument("--gpu-snapshot", help="JSON list of GPU rows from nvidia-smi")
    assess.add_argument("--selected-gpu", type=int)
    assess.add_argument("--gpu-required", action="store_true")
    assess.add_argument("--gpu-query-error")
    args = parser.parse_args(argv)
    payload = assess_node_resources(args.cpu_used_pct, args.ram_used_pct)
    if args.gpu_required or args.selected_gpu is not None or args.gpu_snapshot or args.gpu_query_error:
        rows = None
        query_error = args.gpu_query_error
        if args.gpu_snapshot:
            try:
                rows = json.loads(Path(args.gpu_snapshot).read_text(encoding="utf-8"))
                if not isinstance(rows, list):
                    raise ValueError("GPU snapshot must be a JSON list")
            except (OSError, ValueError) as error:
                rows = []
                query_error = "could not read GPU snapshot: {}".format(error)
        payload = add_gpu_evidence(
            payload, rows, selected_gpu=args.selected_gpu,
            gpu_required=True, query_error=query_error,
        )
    json.dump(payload, fp=sys.stdout, indent=2, sort_keys=True)
    print()
    return 0 if payload["decision"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(_cli())
