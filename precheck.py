"""Finds unsolvable tasks without any model call, from the task text and its files.

Two findings are certain and take the task out of the plan:
- the tests demand two different results from the same call, so no code can pass them all;
- a file or folder the task tells to read is not there.
A third one is only a hint for the agent, because the right answer may still exist (a count of something
that never occurs is zero): a name or quoted term of the task that occurs in none of the files.
"""
import ast
import re
from pathlib import Path

import config

FILE_NAME = re.compile(r"(?<![\w/.-])([\w-]+\.(?:csv|tsv|json|jsonl|txt|md|py|xml|yaml|yml|log|ini|cfg|toml|html|sql))\b")
FOLDER_NAME = re.compile(r"(?<![\w/.-])([\w-]+)/(?=[\s,.;:)]|$)")
# A file named after one of these words is one the task asks to produce, not one it needs.
WRITTEN = re.compile(r"\b(write|writes|written|save|saves|saved|create|creates|created|output|generate|produce|export"
                     r"|store|into|to the file|to a file|named|called)\b[^.]{0,40}$", re.IGNORECASE)
QUOTED = re.compile(r"'([^'\n]{2,40})'|\"([^\"\n]{2,40})\"")
PROPER_NAME = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")
MAX_TEXT_BYTES = 2_000_000


def _files(workdir):
    workdir = Path(workdir)
    if not workdir.is_dir():
        return []
    return [path for path in workdir.rglob("*") if path.is_file() and path.name != "answer.txt"]


def missing_inputs(instruction, workdir):
    """Files and folders the task refers to as given, that are not in the working directory."""
    workdir = Path(workdir)
    have_files = {path.name for path in _files(workdir)}
    have_folders = {path.name for path in workdir.rglob("*") if path.is_dir()} if workdir.is_dir() else set()
    missing = []
    for pattern, have in ((FILE_NAME, have_files), (FOLDER_NAME, have_folders)):
        for match in pattern.finditer(instruction):
            name = match.group(1)
            if name == "answer.txt" or name in have or name in missing:
                continue
            if WRITTEN.search(instruction[:match.start()]):
                continue
            missing.append(name)
    return missing


def _literal(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def _claims(tree):
    """(call, expected result) for every assertion of the form f(constants) == constant, in either order."""
    for node in ast.walk(tree):
        pairs = []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "assertEqual" and len(node.args) >= 2:
                pairs.append((node.args[0], node.args[1]))
            elif node.func.attr in ("assertTrue", "assertFalse") and node.args:
                yield node.args[0], node.func.attr == "assertTrue"
        elif isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
                and isinstance(node.test.ops[0], ast.Eq):
            pairs.append((node.test.left, node.test.comparators[0]))
        for left, right in pairs:
            for call, value in ((left, right), (right, left)):
                if isinstance(call, ast.Call) and _literal(value):
                    yield call, ast.literal_eval(value)
                    break


def contradictory_tests(workdir):
    """Calls that the tests require to return two different results. Only calls with constant arguments count."""
    found = []
    for path in sorted(Path(workdir).glob("test_*.py")) if Path(workdir).is_dir() else []:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        expected = {}
        for call, value in _claims(tree):
            if not all(_literal(arg) for arg in call.args) or not all(_literal(kw.value) for kw in call.keywords):
                continue
            key = ast.dump(call)
            if key in expected and repr(expected[key]) != repr(value):
                text = ast.unparse(call)
                if text not in found:
                    found.append(text)
            expected.setdefault(key, value)
    return found


def absent_terms(instruction, workdir):
    """Quoted terms and proper names of the task that occur in none of its files. A hint, not a verdict."""
    files = _files(workdir)
    if not files:
        return []
    chunks, size = [], 0
    for path in files:
        try:
            data = path.read_bytes()[:MAX_TEXT_BYTES - size]
        except OSError:
            continue
        chunks.append(data.decode("utf-8", errors="replace"))
        size += len(data)
        if size >= MAX_TEXT_BYTES:
            return []  # too much to search fully; a term missing from a part proves nothing
    text = "\n".join(chunks).lower()
    terms = []
    for match in QUOTED.finditer(instruction):
        terms.append(match.group(1) or match.group(2))
    terms += PROPER_NAME.findall(instruction)
    absent = []
    for term in terms:
        term = term.strip()
        if term and term.lower() not in text and term not in absent:
            absent.append(term)
    return absent[:5]


def unsolvable_reason(task):
    """Why the task cannot be solved, or None. Costs no model call."""
    workdir = Path(task.path) / "workspace"
    if config.PRECHECK_CONTRADICTORY_TESTS:
        calls = contradictory_tests(workdir)
        if calls:
            return f"the tests require two different results from {calls[0]}"
    if config.PRECHECK_MISSING_INPUTS:
        missing = missing_inputs(task.instruction, workdir)
        if missing:
            return f"the task needs {', '.join(missing)}, which is not among its files"
    return None
