"""The shell the agent's bash tool runs in, and a start-up check that python3 works there."""
import functools
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Task instructions call python3; on Windows that name is only a Microsoft Store alias, so point it at this Python.
PYTHON3_PREAMBLE = f'python3() {{ "{Path(sys.executable).as_posix()}" "$@"; }}; ' if os.name == "nt" else ""


@functools.lru_cache(maxsize=None)
def find_bash():
    """Path to bash. On Windows this must be Git's bash: System32\\bash.exe is WSL, a separate Linux
    that has neither this Python nor the Windows paths of the tasks."""
    if os.name != "nt":
        return shutil.which("bash") or "/bin/bash"
    candidates = []
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent  # ...\Git\cmd\git.exe -> ...\Git
        candidates += [git_root / "bin" / "bash.exe", git_root / "usr" / "bin" / "bash.exe"]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            candidates.append(Path(base) / "Git" / "bin" / "bash.exe")
    on_path = shutil.which("bash")
    if on_path and "system32" not in on_path.lower():
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Git Bash not found. Install Git for Windows: System32\\bash.exe is WSL and cannot run "
                       "this Python.")


def check():
    """Run python3 once through the agent's bash. Raises RuntimeError when it does not work."""
    bash = find_bash()
    result = subprocess.run([bash, "-c", PYTHON3_PREAMBLE + 'python3 -c "print(6 * 7)"'], capture_output=True,
                            text=True, timeout=60)
    if result.returncode != 0 or result.stdout.strip() != "42":
        raise RuntimeError(f"python3 does not work in {bash}: exit {result.returncode}, "
                           f"output {(result.stdout + result.stderr).strip()[:300]!r}")
    log.info("agent shell %s, python3 works", bash)
    return bash
