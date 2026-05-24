from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    allowed = []
    for ch in str(text):
        allowed.append(ch if ch.isalnum() or ch in "._=-" else "-")
    return "".join(allowed).strip("-")[:160] or "unnamed"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def tail_text(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def wait_ready(url: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def terminate_tree(proc: subprocess.Popen[Any], grace_sec: int = 20) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_capture(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.strip()
    except Exception as exc:
        return 999, str(exc)


def detect_oom(text: str) -> bool:
    t = text.lower()
    needles = ["out of memory", "cuda oom", "cublas_status_alloc_failed", "torch.cuda.outofmemoryerror"]
    return any(n in t for n in needles)
