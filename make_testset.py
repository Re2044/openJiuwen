"""Generates a test set shaped like the final one: tasks of the four example types, part of them unsolvable.

Usage: python make_testset.py [out_dir] [--count 100] [--unsolvable 0.2] [--seed 2026] [--force]

Every task folder looks like the examples: task.json plus, where needed, workspace/.
Fields used only for scoring, never by the solver:
- expected: the reference answer, as in the examples
- solvable: false for the unsolvable tasks; the right move there is to answer nothing
- hidden_tests: for coding tasks, extra tests run at scoring time, like the final hidden tests
The generator checks itself: every solvable task has a correct answer, every unsolvable one has none.
"""
import argparse
import csv
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TAIL = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODING_INSTRUCTION = (
    "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit "
    "the tests) until all tests pass. Work only inside the current directory. When you are done, write ONLY the word "
    "DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
TYPES = ("math", "data_analysis", "search", "coding")


class Task:
    def __init__(self, task_type, instruction, expected, check, solvable=True, files=None, hidden_tests=None):
        self.task_type = task_type
        self.instruction = instruction
        self.expected = expected
        self.check = check
        self.solvable = solvable
        self.files = files or {}  # path inside workspace/ -> text
        self.hidden_tests = hidden_tests


# ---------------------------------------------------------------- math

def make_math(rng, solvable):
    stock = start = rng.randint(300, 3000)
    sentences = []
    steps = rng.randint(2, 6)
    broken_step = rng.randrange(steps) if not solvable else None
    for step in range(steps):
        kind = rng.choice(["ship", "receive", "split", "double"])
        if step == broken_step:
            if kind in ("split", "double"):
                sentences.append("Then the stock is split equally into 0 warehouses and it keeps one share "
                                 "(rounded down).")
            elif kind == "ship":
                sentences.append(f"It ships some boxes per day for {rng.randint(2, 6)} days.")
            else:
                sentences.append(f"It receives several deliveries of {rng.randint(20, 150)} boxes each.")
            continue
        if kind == "ship" and stock > 50:
            days = rng.randint(2, 6)
            per_day = rng.randint(5, max(6, stock // (days * 2)))
            stock -= per_day * days
            sentences.append(f"It ships {per_day} boxes per day for {days} days.")
        elif kind == "receive":
            count, size = rng.randint(2, 5), rng.randint(20, 150)
            stock += count * size
            sentences.append(f"It receives {count} deliveries of {size} boxes each.")
        elif kind == "split" and stock > 20:
            parts = rng.randint(2, 4)
            stock //= parts
            sentences.append(f"Then the stock is split equally into {parts} warehouses and it keeps one share "
                             "(rounded down).")
        else:
            stock *= 2
            sentences.append("Then the stock is doubled by merging with a sister warehouse.")
    instruction = (f"A warehouse starts with {start} boxes. {' '.join(sentences)} How many boxes does it have at the "
                   f"end? Compute it with the bash tool (python3 -c). {TAIL}")
    return Task("math", instruction, str(stock) if solvable else None, "exact_int", solvable)


# ---------------------------------------------------------------- data_analysis

REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["widget", "gadget", "sprocket", "gizmo"]


def make_sales(rng, rows):
    data = [{"order_id": i + 1, "region": rng.choice(REGIONS), "product": rng.choice(PRODUCTS),
             "month": rng.randint(1, 12), "units": rng.randint(1, 40), "price": rng.randint(5, 99)}
            for i in range(rows)]
    data.sort(key=lambda row: (row["month"], row["order_id"]))
    for i, row in enumerate(data, start=1):
        row["order_id"] = i
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["order_id", "region", "product", "month", "units", "price"],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(data)
    return data, out.getvalue()


def make_data_analysis(rng, solvable):
    rows = rng.choice([20, 50, 100, 300])
    data, text = make_sales(rng, rows)
    head = f"The file sales.csv has columns order_id,region,product,month,units,price ({rows} rows)."
    product = rng.choice(PRODUCTS)
    if not solvable:
        if rng.random() < 0.5:
            question = f"What is the total discount given on product '{product}'?"
            files = {"sales.csv": text}
        else:  # the data file the task talks about is not there
            question = f"How many units of product '{product}' were sold in region '{rng.choice(REGIONS)}'?"
            files = {"notes.txt": "The sales export has not been delivered yet.\n"}
        return Task("data_analysis", f"{head} {question} Use bash with python3 to compute it. {TAIL}", None,
                    "exact_int", False, files)
    while True:
        kind = rng.choice(["months", "region", "revenue"])
        if kind == "months":
            first = rng.randint(1, 11)
            last = rng.randint(first, min(12, first + 3))
            answer = sum(r["units"] for r in data if r["product"] == product and first <= r["month"] <= last)
            question = (f"How many units of product '{product}' were sold in months {first} through {last} "
                        "inclusive?")
        elif kind == "region":
            region = rng.choice(REGIONS)
            answer = sum(r["units"] for r in data if r["product"] == product and r["region"] == region)
            question = f"How many units of product '{product}' were sold in region '{region}'?"
        else:
            month = rng.randint(1, 12)
            answer = sum(r["units"] * r["price"] for r in data if r["product"] == product and r["month"] == month)
            question = f"What is the total revenue (units times price) of product '{product}' in month {month}?"
        if answer > 0:
            break
    return Task("data_analysis", f"{head} {question} Use bash with python3 to compute it. {TAIL}", str(answer),
                "exact_int", True, {"sales.csv": text})


# ---------------------------------------------------------------- search

FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Lena",
         "Maya", "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya", "Umar", "Vera", "Wei", "Yusuf"]
LAST = ["Smith", "Novak", "Haddad", "Wang", "Kowalski", "Petrov", "Sato", "Garcia", "Muller", "Ivanova", "Larsen",
        "Okafor", "Rossi", "Kim"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods", "Kestrel Cloud", "Lumen Grid",
             "Mosaic Health", "Nimbus Freight", "Onyx Security", "Pinecrest Media"]
CITIES = ["Tbilisi", "Osaka", "Lisbon", "Nairobi", "Oslo", "Lima", "Hanoi", "Tallinn", "Porto", "Seoul", "Austin",
          "Krakow", "Dublin", "Quito", "Riga", "Perth"]
SEARCH_HEAD = "The directory kb/ contains a small knowledge base of markdown files."
SEARCH_TAIL = f"Use the file tools or bash (ls, grep, cat) to find it. {TAIL}"


def slug(name):
    return name.lower().replace(" ", "_")


def make_search(rng, solvable):
    companies = {name: {"Headquarters": rng.choice(CITIES), "Founded": rng.randint(1950, 2020)}
                 for name in rng.sample(COMPANIES, 10)}
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 30)
    people = {name: {"Born": rng.randint(1950, 2002), "Employer": rng.choice(list(companies)),
                     "Mentor": rng.choice([n for n in names if n != name])} for name in names}
    hidden_company = None  # employer whose file is left out

    kind = rng.choice(["born", "hq", "founded", "mentor_hq"])
    person = rng.choice(names)
    if solvable:
        employer = people[person]["Employer"]
        mentor_employer = people[people[person]["Mentor"]]["Employer"]
        expected = {"born": str(people[person]["Born"]), "hq": companies[employer]["Headquarters"],
                    "founded": str(companies[employer]["Founded"]),
                    "mentor_hq": companies[mentor_employer]["Headquarters"]}[kind]
    else:
        expected = None
        flaw = rng.choice(["no_person", "no_field", "no_company"])
        if flaw == "no_person":
            person = rng.choice([f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" not in people])
        elif flaw == "no_field":
            field = "Born" if kind == "born" else "Employer"
            if kind == "mentor_hq":
                del people[people[person]["Mentor"]]["Employer"]
            else:
                del people[person][field]
        else:
            if kind == "born":
                kind = "hq"
            target = people[person]["Employer"] if kind != "mentor_hq" else people[people[person]["Mentor"]]["Employer"]
            hidden_company = target
            # Nobody else may point to the missing company with the same question, so keep the kb consistent.
    question = {"born": f"In which year was {person} born?",
                "hq": f"In which city is the headquarters of {person}'s employer?",
                "founded": f"In which year was {person}'s employer founded?",
                "mentor_hq": f"In which city is the headquarters of the employer of {person}'s mentor?"}[kind]

    files = {}
    for name, info in companies.items():
        if name != hidden_company:
            files[f"kb/company_{slug(name)}.md"] = (f"# {name}\n\nHeadquarters: {info['Headquarters']}\n"
                                                    f"Founded: {info['Founded']}\n")
    for name, info in people.items():
        lines = [f"{key}: {info[key]}" for key in ("Born", "Employer", "Mentor") if key in info]
        files[f"kb/person_{slug(name)}.md"] = f"# {name}\n\n" + "\n".join(lines) + "\n"
    return Task("search", f"{SEARCH_HEAD} {question} {SEARCH_TAIL}", expected, "exact_str", solvable, files)


# ---------------------------------------------------------------- coding

FUNCTIONS = {
    "reverse_words": {
        "correct": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
        "bugs": ["def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                 "def reverse_words(s):\n    return ' '.join(s.split())\n"],
        "visible": ["self.assertEqual(m.reverse_words('c a b'), 'b a c')"],
        "hidden": ["self.assertEqual(m.reverse_words('one'), 'one')",
                   "self.assertEqual(m.reverse_words('x  y'), 'y x')"],
        "contradiction": "self.assertEqual(m.reverse_words('c a b'), 'c a b')",
    },
    "total": {
        "correct": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
        "bugs": ["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
                 "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n"],
        "visible": ["self.assertEqual(m.total([1, 2, 3]), 6)"],
        "hidden": ["self.assertEqual(m.total([]), 0)", "self.assertEqual(m.total([-1, 5]), 4)"],
        "contradiction": "self.assertEqual(m.total([1, 2, 3]), 7)",
    },
    "maximum": {
        "correct": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n"
                   "    return best\n",
        "bugs": ["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n"
                 "    return best\n",
                 "def maximum(xs):\n    best = xs[0]\n    for x in xs[:-1]:\n        if x > best:\n"
                 "            best = x\n    return best\n"],
        "visible": ["self.assertEqual(m.maximum([3, 9, 2]), 9)", "self.assertEqual(m.maximum([1, 5]), 5)"],
        "hidden": ["self.assertEqual(m.maximum([-5, -2, -9]), -2)", "self.assertEqual(m.maximum([7]), 7)"],
        "contradiction": "self.assertEqual(m.maximum([3, 9, 2]), 3)",
    },
    "minimum": {
        "correct": "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n"
                   "    return best\n",
        "bugs": ["def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n"
                 "    return best\n"],
        "visible": ["self.assertEqual(m.minimum([4, 1, 7]), 1)"],
        "hidden": ["self.assertEqual(m.minimum([-3, -8]), -8)", "self.assertEqual(m.minimum([2]), 2)"],
        "contradiction": "self.assertEqual(m.minimum([4, 1, 7]), 4)",
    },
    "average": {
        "correct": "def average(xs):\n    return sum(xs) / len(xs)\n",
        "bugs": ["def average(xs):\n    return sum(xs) / (len(xs) + 1)\n",
                 "def average(xs):\n    return sum(xs) // len(xs)\n"],
        "visible": ["self.assertEqual(m.average([2, 4, 6]), 4)", "self.assertEqual(m.average([1, 2]), 1.5)"],
        "hidden": ["self.assertEqual(m.average([5]), 5)", "self.assertEqual(m.average([-1, 1]), 0)"],
        "contradiction": "self.assertEqual(m.average([2, 4, 6]), 5)",
    },
    "first_n": {
        "correct": "def first_n(xs, n):\n    return xs[:n]\n",
        "bugs": ["def first_n(xs, n):\n    return xs[:n - 1]\n", "def first_n(xs, n):\n    return xs[1:n + 1]\n"],
        "visible": ["self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"],
        "hidden": ["self.assertEqual(m.first_n([1, 2], 5), [1, 2])", "self.assertEqual(m.first_n([1, 2, 3], 0), [])"],
        "contradiction": "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6])",
    },
    "count_even": {
        "correct": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
        "bugs": ["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
                 "def count_even(xs):\n    return sum(1 for x in xs if x % 2)\n"],
        "visible": ["self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"],
        "hidden": ["self.assertEqual(m.count_even([]), 0)", "self.assertEqual(m.count_even([0, -2, 3]), 2)"],
        "contradiction": "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 2)",
    },
    "is_palindrome": {
        "correct": "def is_palindrome(s):\n    return s == s[::-1]\n",
        "bugs": ["def is_palindrome(s):\n    return s == s[::1]\n",
                 "def is_palindrome(s):\n    return s[:1] == s[-1:]\n"],
        "visible": ["self.assertTrue(m.is_palindrome('racecar'))", "self.assertFalse(m.is_palindrome('abca'))"],
        "hidden": ["self.assertTrue(m.is_palindrome(''))", "self.assertFalse(m.is_palindrome('ab'))"],
        "contradiction": "self.assertTrue(m.is_palindrome('abca'))",
    },
    "factorial": {
        "correct": "def factorial(n):\n    result = 1\n    for k in range(2, n + 1):\n        result *= k\n"
                   "    return result\n",
        "bugs": ["def factorial(n):\n    result = 1\n    for k in range(2, n):\n        result *= k\n"
                 "    return result\n",
                 "def factorial(n):\n    result = 0\n    for k in range(2, n + 1):\n        result *= k\n"
                 "    return result\n"],
        "visible": ["self.assertEqual(m.factorial(5), 120)"],
        "hidden": ["self.assertEqual(m.factorial(0), 1)", "self.assertEqual(m.factorial(6), 720)"],
        "contradiction": "self.assertEqual(m.factorial(5), 24)",
    },
    "clamp": {
        "correct": "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
        "bugs": ["def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
                 "def clamp(x, lo, hi):\n    return max(lo, min(x, lo))\n"],
        "visible": ["self.assertEqual(m.clamp(5, 0, 10), 5)", "self.assertEqual(m.clamp(-3, 0, 10), 0)",
                    "self.assertEqual(m.clamp(15, 0, 10), 10)"],
        "hidden": ["self.assertEqual(m.clamp(10, 0, 10), 10)", "self.assertEqual(m.clamp(0, 0, 0), 0)"],
        "contradiction": "self.assertEqual(m.clamp(5, 0, 10), 10)",
    },
    "count_vowels": {
        "correct": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "bugs": ["def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
                 "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeio')\n"],
        "visible": ["self.assertEqual(m.count_vowels('Apple'), 2)", "self.assertEqual(m.count_vowels('queue'), 4)"],
        "hidden": ["self.assertEqual(m.count_vowels(''), 0)", "self.assertEqual(m.count_vowels('xyz'), 0)"],
        "contradiction": "self.assertEqual(m.count_vowels('Apple'), 1)",
    },
    "unique_sorted": {
        "correct": "def unique_sorted(xs):\n    return sorted(set(xs))\n",
        "bugs": ["def unique_sorted(xs):\n    return sorted(xs)\n",
                 "def unique_sorted(xs):\n    return sorted(set(xs), reverse=True)\n"],
        "visible": ["self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3])"],
        "hidden": ["self.assertEqual(m.unique_sorted([]), [])", "self.assertEqual(m.unique_sorted([5, 5]), [5])"],
        "contradiction": "self.assertEqual(m.unique_sorted([3, 1, 3, 2]), [1, 2, 3, 3])",
    },
}


def test_module(methods):
    body = "\n".join(f"    def test_{name}(self):\n" + "".join(f"        {line}\n" for line in lines)
                     for name, lines in methods)
    return ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n" + body +
            "\n\nif __name__ == '__main__':\n    unittest.main()\n")


def library(names, chosen_bugs):
    return "\n\n".join(chosen_bugs.get(name, FUNCTIONS[name]["correct"]) for name in names)


def make_coding(rng, solvable):
    names = rng.sample(list(FUNCTIONS), rng.randint(4, 7))
    bugged = rng.sample(names, rng.randint(1, min(3, len(names))))
    bugs = {name: rng.choice(FUNCTIONS[name]["bugs"]) for name in bugged}
    methods = [(name, FUNCTIONS[name]["visible"]) for name in names]
    if not solvable:
        # Two tests demand different results for the same input: no mylib.py can pass both.
        target = rng.choice(names)
        methods.append((f"{target}_spec", [FUNCTIONS[target]["contradiction"]]))
    rng.shuffle(methods)
    hidden = test_module([(f"{name}_hidden", FUNCTIONS[name]["hidden"]) for name in names])
    files = {"mylib.py": library(names, bugs), "test_mylib.py": test_module(methods)}
    task = Task("coding", CODING_INSTRUCTION, "tests_pass", "unittest", solvable, files, hidden)
    task.reference = library(names, {})  # the fixed library, used only by the self-check
    task.single_bug_libraries = [library(names, {name: bugs[name]}) for name in bugged]
    return task


def run_tests(files, extra_test=None):
    """True when every test file passes with the given workspace files."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items():
            Path(tmp, name).write_text(text, encoding="utf-8")
        if extra_test:
            Path(tmp, "test_hidden.py").write_text(extra_test, encoding="utf-8")
        tests = ["test_mylib.py"] + (["test_hidden.py"] if extra_test else [])
        return all(subprocess.run([sys.executable, t], cwd=tmp, capture_output=True, timeout=60).returncode == 0
                   for t in tests)


def check_coding(task):
    """Buggy code fails; each single bug is caught by the visible tests; the fix passes only when solvable."""
    assert not run_tests(task.files), "buggy library passes"
    for buggy in task.single_bug_libraries:
        assert not run_tests({**task.files, "mylib.py": buggy}), "a bug is not caught by the visible tests"
    fixed = {**task.files, "mylib.py": task.reference}
    if task.solvable:
        assert run_tests(fixed, task.hidden_tests), "the reference fix fails"
    else:
        assert not run_tests(fixed), "the contradictory tests can be passed"


# ---------------------------------------------------------------- set

MAKERS = {"math": make_math, "data_analysis": make_data_analysis, "search": make_search, "coding": make_coding}


def build(count, unsolvable_share, seed):
    rng = random.Random(seed)
    types = [TYPES[i % len(TYPES)] for i in range(count)]
    rng.shuffle(types)
    unsolvable = set(rng.sample(range(count), round(count * unsolvable_share)))
    tasks = []
    for index, task_type in enumerate(types):
        task = MAKERS[task_type](rng, index not in unsolvable)
        task.task_id = f"{task_type}_{index:04d}"
        task.reward = rng.randint(1, 5)  # random and unrelated to difficulty, as in the rules
        if task_type == "coding":
            check_coding(task)
        tasks.append(task)
    return tasks


def write(tasks, out_dir):
    for task in tasks:
        folder = out_dir / task.task_id
        folder.mkdir(parents=True)
        meta = {"task_id": task.task_id, "task_type": task.task_type, "instruction": task.instruction,
                "reward": task.reward, "check": task.check}
        if task.expected is not None:
            meta["expected"] = task.expected
        if not task.solvable:
            meta["solvable"] = False
        if task.hidden_tests:
            meta["hidden_tests"] = task.hidden_tests
        meta.update(getattr(task, "extra", {}))  # labels added by make_suites.py, e.g. suite and difficulty
        (folder / "task.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name, text in task.files.items():
            path = folder / "workspace" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", default="task-dir-final", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--unsolvable", type=float, default=0.2, help="share of unsolvable tasks (rules: 0.15-0.25)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="replace an existing out_dir")
    args = parser.parse_args()
    if args.out_dir.exists():
        if not args.force:
            sys.exit(f"{args.out_dir} exists; pass --force to replace it")
        shutil.rmtree(args.out_dir)

    tasks = build(args.count, args.unsolvable, args.seed)
    write(tasks, args.out_dir)
    print(f"{len(tasks)} tasks in {args.out_dir}, seed {args.seed}")
    for task_type in TYPES:
        of_type = [t for t in tasks if t.task_type == task_type]
        print(f"  {task_type:<14} {len(of_type):>3} tasks, {sum(not t.solvable for t in of_type):>2} unsolvable")
    unsolvable = [t for t in tasks if not t.solvable]
    print(f"  unsolvable     {len(unsolvable)} ({len(unsolvable) / len(tasks):.0%}); "
          f"reachable reward {sum(t.reward for t in tasks if t.solvable)} of {sum(t.reward for t in tasks)}")


if __name__ == "__main__":
    main()
