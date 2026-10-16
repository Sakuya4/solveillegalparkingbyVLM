from contextlib import contextmanager
import json
import os
from pathlib import Path

import psutil


def active_research_child(root: Path) -> int | None:
    for path in (root / "outputs/research_runs").glob("*status.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        pid = state.get("child_pid")
        if not pid:
            continue
        try:
            command = psutil.Process(pid).cmdline()
        except psutil.NoSuchProcess:
            continue
        expected = Path(state["command"][1]).name
        if any(Path(part).name == expected for part in command[1:]):
            return pid
    return None


@contextmanager
def exclusive_gpu_job(root: Path):
    """OS-owned locks are released even if a worker is terminated."""
    directory = root / "outputs/research_runs"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "gpu_workload.lock").open("a+b") as handle:
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another GPU job is active; retry after it completes") from exc
        try:
            pid = active_research_child(root)
            if pid is not None:
                raise RuntimeError(f"Research child {pid} is active; inspect its log before GPU work")
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
