from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = "qdev2d_velocity_gradient_float64_v1"


def process_memory():
    """OS process peak working set; includes imports, model and mapped pages."""
    import os
    if os.name != "nt":
        import resource
        factor = 1 if __import__("sys").platform == "darwin" else 1024
        return {"peak_resident_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor)}
    import ctypes as c
    from ctypes import wintypes as w
    class Counters(c.Structure):
        _fields_ = [("cb",w.DWORD),("PageFaultCount",w.DWORD)] + [(name,c.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage")]
    counters = Counters(); counters.cb = c.sizeof(counters)
    kernel=c.WinDLL("kernel32",use_last_error=True); kernel.GetCurrentProcess.restype=w.HANDLE
    psapi=c.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[w.HANDLE,c.POINTER(Counters),w.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),c.byref(counters),counters.cb):
        return {"peak_resident_bytes":None,"error":c.get_last_error()}
    return {"peak_resident_bytes":int(counters.PeakWorkingSetSize),"peak_pagefile_bytes":int(counters.PeakPagefileUsage),"scope":"entire Python process, including imports and mapped patch pages"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_groups(records: list[dict]) -> None:
    seen: dict[str, str] = {}
    ids: set[str] = set()
    for record in records:
        group, split = record["leakage_group_id"], record["split"]
        if group in seen and seen[group] != split:
            raise ValueError(f"Trajectory leakage: {group}")
        if record["dataset_id"] in ids:
            raise ValueError("Duplicate dataset ID")
        ids.add(record["dataset_id"])
        seen[group] = split
