"""A short, factual description of a task's workspace for the judge: what files there are and how big they are.

The judge only sees the instruction otherwise, and the instruction of a hard task often reads exactly like that of
an easy one: the difference is in the files. The summary is made by this script, costs no model call, and is kept
under config.WORKSPACE_SUMMARY_MAX_CHARS (about a tenth of that in cost units, since input counts / 10).

What it reports:
- file count and total size, grouped by folder when a folder holds many files, with one sample file;
- CSV files: rows, columns with their value ranges or distinct values, repeated ids, non-numeric values in numeric
  columns;
- Python files: lines and function names; for test files, the outcome of running them on a copy (how many tests
  fail, and which);
- the full text of very small files (notes, pointers to other files);
- files and folders the instruction names that are not there.
It ends with "Warnings" for signs that the data is incomplete or withheld, which the judge should weigh in its
solvable_probability: files named in the instruction or in other files that do not exist, non-numeric values in
numeric CSV columns, and small text files that say the data is withheld (restricted, redacted, not available, ...).
"""
import csv
import io
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import config

SMALL_FILE_CHARS = 300  # files up to this size are shown in full
MAX_FILES_LISTED = 5  # a folder with more files is summarised as a group
MAX_DISTINCT_SHOWN = 8
NAMED_FILE = re.compile(r"(?<![\w/.-])((?:[\w-]+/)*[\w-]+\.(?:csv|tsv|txt|md|json|py|log|xlsx?))\b")
WITHHELD = re.compile(r"\b(restricted|redacted|not (?:yet )?(?:available|included|delivered|published|stored)|"
                      r"pending|encrypted|confidential|superseded|do not use|has not been)\b", re.IGNORECASE)
NAMED_DIR = re.compile(r"(?<![\w/.-])([\w-]+)/(?=[\s,.;:)]|$)")
SKIP = {"answer.txt"}


def _files(workspace):
    return sorted(p for p in workspace.rglob("*") if p.is_file() and p.name not in SKIP
                  and "__pycache__" not in p.parts) if workspace.is_dir() else []


def warnings(task_path, instruction=""):
    """Only the warnings: signs that the data the task needs is missing or withheld. Empty when there are none."""
    workspace = Path(task_path) / "workspace"
    return _warnings(instruction, workspace, _files(workspace))


def summarize(task_path, instruction=""):
    """Summary text for the task folder at task_path (its workspace/ subfolder is what the agent sees)."""
    workspace = Path(task_path) / "workspace"
    files = _files(workspace)
    lines = []
    if not files:
        lines.append("No files: the agent starts in an empty folder.")
    else:
        total = sum(p.stat().st_size for p in files)
        lines.append(f"{len(files)} file{'s' * (len(files) != 1)}, {_size(total)} in total.")
        by_folder = defaultdict(list)
        for path in files:
            by_folder[path.parent.relative_to(workspace).as_posix()].append(path)
        for folder, members in sorted(by_folder.items()):
            if len(members) > MAX_FILES_LISTED:
                lines.append(_group(folder, members, workspace))
            else:
                lines += [_describe(p, workspace) for p in members]
        tests = [p for p in files if p.parent == workspace and p.name.startswith("test_") and p.suffix == ".py"]
        if tests and config.JUDGE_RUN_TESTS:
            lines.append(_run_tests(workspace, tests))
    warnings = _warnings(instruction, workspace, files)
    if warnings:
        lines.append("Warnings: " + " ".join(warnings))
    text = "\n".join(lines)
    limit = config.WORKSPACE_SUMMARY_MAX_CHARS
    return text if len(text) <= limit else text[: limit - 20].rstrip() + "\n[summary cut]"


def _size(n):
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def _read(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _describe(path, workspace):
    name = path.relative_to(workspace).as_posix()
    text = _read(path)
    if text is None:
        return f"- {name}: binary, {_size(path.stat().st_size)}"
    if path.suffix in (".csv", ".tsv"):
        return f"- {name}: {_csv(text, path.suffix)}"
    if path.suffix == ".py":
        return f"- {name}: {_python(text, test=path.name.startswith('test_'))}"
    n_lines = text.count("\n") + (not text.endswith("\n") and bool(text))
    if len(text) <= SMALL_FILE_CHARS:
        return f"- {name}: {n_lines} line{'s' * (n_lines != 1)}: {text.strip()!r}"
    longest = max((len(line) for line in text.splitlines()), default=0)
    if longest > 1000 and " " not in text[:1000]:
        return f"- {name}: {len(text)} chars on {n_lines} line(s) without spaces (looks encoded)"
    return f"- {name}: {n_lines} lines, {len(text)} chars"


def _group(folder, members, workspace):
    suffixes = sorted({p.suffix or "(none)" for p in members})
    texts = [t for t in (_read(p) for p in members) if t is not None]
    chars = sum(len(t) for t in texts)
    avg_lines = sum(t.count("\n") for t in texts) / max(len(texts), 1)
    prefixes = defaultdict(int)
    for p in members:
        prefixes[p.name.split("_")[0] if "_" in p.name else p.stem] += 1
    kinds = ", ".join(f"{count} {prefix}_*" for prefix, count in sorted(prefixes.items(), key=lambda kv: -kv[1])[:4])
    sample = min(members, key=lambda p: p.name)
    sample_text = (_read(sample) or "").strip()
    sample_text = sample_text[:200] + ("..." if len(sample_text) > 200 else "")
    return (f"- {folder}/: {len(members)} files ({', '.join(suffixes)}; {kinds}), {chars} chars, about "
            f"{avg_lines:.0f} lines each. Sample {sample.name}: {sample_text!r}")


def _csv(text, suffix):
    reader = csv.reader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ",")
    rows = list(reader)
    if not rows:
        return "empty"
    header, data = rows[0], rows[1:]
    parts = []
    for i, column in enumerate(header):
        values = [r[i].strip() if i < len(r) else "" for r in data]
        numeric = [v for v in values if re.fullmatch(r"-?\d+(\.\d+)?", v)]
        if numeric and len(numeric) >= len(values) / 2:
            nums = [float(v) for v in numeric]
            desc = f"{column} (number {min(nums):g}-{max(nums):g}"
            other = sorted({v for v in values if v not in numeric})
            if other:
                desc += f"; {len(values) - len(numeric)} non-numeric such as {other[0]!r}"
            parts.append(desc + ")")
            continue
        raw = sorted({(r[i] if i < len(r) else "") for r in data})
        shown = ", ".join(repr(v) for v in raw[:MAX_DISTINCT_SHOWN]) + (", ..." if len(raw) > MAX_DISTINCT_SHOWN else "")
        desc = f"{column} (text, {len(raw)} distinct: {shown}"
        if len({v.strip().lower() for v in raw}) < len(raw):
            desc += "; some differ only in case or spaces"
        parts.append(desc + ")")
    repeated = len(data) - len({r[0] for r in data if r})
    tail = f"; {repeated} rows repeat an earlier {header[0]}" if repeated else ""
    return f"{len(data)} data rows, columns: " + ", ".join(parts) + tail


def _python(text, test):
    n_lines = text.count("\n")
    if test:
        tests = re.findall(r"^\s+def (test_\w+)", text, re.M)
        return f"{n_lines} lines, {len(tests)} test methods"
    functions = re.findall(r"^def (\w+)", text, re.M)
    shown = ", ".join(functions[:10]) + (", ..." if len(functions) > 10 else "")
    return f"{n_lines} lines, {len(functions)} functions: {shown}"


def _run_tests(workspace, tests):
    """Run the visible tests on a copy of the workspace; the task folder itself is never touched."""
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "ws"
        shutil.copytree(workspace, copy, ignore=shutil.ignore_patterns("answer.txt", "__pycache__"))
        for test in tests:
            try:
                run = subprocess.run([sys.executable, test.name], cwd=copy, capture_output=True, text=True,
                                     timeout=config.JUDGE_TEST_TIMEOUT_SECONDS,
                                     env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": ""})
            except subprocess.TimeoutExpired:
                results.append(f"{test.name}: timed out")
                continue
            output = run.stdout + run.stderr
            ran = re.search(r"Ran (\d+) test", output)
            failing = sorted(set(re.findall(r"^(?:FAIL|ERROR): (test_\w+)", output, re.M)))
            if ran is None:
                results.append(f"{test.name}: does not run ({(output.strip().splitlines() or ['no output'])[-1][:120]})")
            elif failing:
                results.append(f"{test.name}: {len(failing)} of {ran.group(1)} tests fail ({', '.join(failing[:8])})")
            else:
                results.append(f"{test.name}: all {ran.group(1)} tests pass")
    return "Visible tests now: " + "; ".join(results) + "."


def _missing(text, workspace, base=None, dirs=True):
    """Files (and folders) named in text that do not exist, relative to base or the workspace root."""
    named = set(NAMED_FILE.findall(text)) | ({d + "/" for d in NAMED_DIR.findall(text)} if dirs else set())
    named -= {"answer.txt", "task.json"}
    return sorted(n for n in named if not (workspace / n.rstrip("/")).exists()
                  and not (base is not None and (base / n.rstrip("/")).exists()))


def _warnings(instruction, workspace, files):
    warnings = []
    missing = _missing(instruction, workspace)
    if missing:
        warnings.append(f"the instruction names {', '.join(missing)}, which is not in the workspace.")
    for path in files:
        text = _read(path)
        if text is None:
            continue
        name = path.relative_to(workspace).as_posix()
        if path.suffix in (".csv", ".tsv"):
            for match in re.finditer(r"(\w+) \(number [^;)]*; (\d+) non-numeric such as ('[^']*')", _csv(text, path.suffix)):
                warnings.append(f"column {match.group(1)} of {name} has {match.group(2)} values that are not numbers, "
                                f"such as {match.group(3)}.")
        elif len(text) <= 2000:
            gone = _missing(text, workspace, base=path.parent, dirs=False)
            if gone:
                warnings.append(f"{name} names {', '.join(gone)}, which is not in the workspace.")
            said = WITHHELD.search(text)
            if said:
                warnings.append(f"{name} says {said.group(0)!r}.")
    return warnings


if __name__ == "__main__":  # python workspace_summary.py <task folder>...: print the summaries
    import json

    for folder in sys.argv[1:]:
        meta = json.loads((Path(folder) / "task.json").read_text(encoding="utf-8"))
        print(f"== {meta['task_id']}\n{summarize(folder, meta['instruction'])}\n")
