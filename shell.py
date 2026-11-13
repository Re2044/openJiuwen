"""The shell the agent's bash tool runs in, and a start-up check that python3 works there."""
import asyncio
import functools
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Task instructions call python3; on Windows that name is only a Microsoft Store alias, so point it at this Python.
PYTHON3_PREAMBLE = f'python3() {{ "{Path(sys.executable).as_posix()}" "$@"; }}; ' if os.name == "nt" else ""
KILL_GRACE_SECONDS = 5  # how long to wait for a killed command to go away


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


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)


def _attach_job(proc):
    """Windows: put the command into a new Job Object and return its handle, or None.

    Programs started by Git Bash are not always children of the shell in Windows' eyes, so `taskkill /T`
    misses them; every process started inside a job belongs to the job and ends with it.
    """
    if os.name != "nt":
        return None
    try:
        popen = proc._transport.get_extra_info("subprocess")
        job = _kernel32.CreateJobObjectW(None, None)
        if job and _kernel32.AssignProcessToJobObject(job, int(popen._handle)):
            return job
        if job:
            _kernel32.CloseHandle(job)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def kill_process_tree(proc, job=None):
    """Stop a command together with everything it started.

    Killing only the shell leaves its children running (a `find /`, a python loop), and they keep the
    output pipe open, so the caller would go on waiting for them.
    """
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)  # the command runs in its own session, see run_command
        elif job:
            _kernel32.TerminateJobObject(job, 1)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


async def run_command(argv, cwd, timeout):
    """Run a command; the return code is None on timeout. The proxy key is not passed to it."""
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "LLM_BASE_URL")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # POSIX: its own session, so that a timeout can kill the whole process group. Windows: a job, see above.
    group = {} if os.name == "nt" else {"start_new_session": True}
    proc = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                **group)
    job = _attach_job(proc)
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        kill_process_tree(proc, job)
        try:
            await asyncio.wait_for(proc.wait(), KILL_GRACE_SECONDS)
        except asyncio.TimeoutError:
            log.warning("command did not exit %ss after being killed: %s", KILL_GRACE_SECONDS, argv[-1][-80:])
        return None, f"timed out after {timeout}s; the command and everything it started were stopped"
    except asyncio.CancelledError:
        kill_process_tree(proc, job)
        raise
    finally:
        if job:
            _kernel32.CloseHandle(job)
    return proc.returncode, stdout.decode("utf-8", errors="replace")
