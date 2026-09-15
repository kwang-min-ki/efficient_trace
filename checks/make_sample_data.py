"""다운로드 없는 소규모 math/code 평가 JSONL·학습 parquet 생성"""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data as D

# Big-Math-style: hard competition problems with integer answers.
MATH = [
    ("Let $N$ be the number of ordered pairs $(a,b)$ of positive integers with "
     "$a\\le b\\le 60$ such that $ab$ is divisible by $a+b$. Find $N$.", "61"),
    ("Find the least positive integer $n$ such that $n!$ ends in exactly 100 zeros.", "405"),
    ("A sequence satisfies $a_1=1$ and $a_{n+1}=a_n+\\gcd(n,a_n)$. Find $a_{20}$.", "21"),
    ("How many subsets of $\\{1,2,\\dots,12\\}$ contain no two consecutive integers?", "377"),
    ("Compute the remainder when $3^{2024}$ is divided by $1000$.", "481"),
    ("Let $f(x)=x^3-3x+1$. How many real $x$ satisfy $f(f(x))=0$?", "7"),
    # Tractable problems help responses reach </think> to exercise scoring.
    ("What is the sum of the first 30 positive integers?", "465"),
    ("Compute $7^3 - 4^3$.", "279"),
    ("How many positive divisors does $360$ have?", "24"),
    ("What is $\\gcd(84, 126)$?", "42"),
    ("A rectangle has perimeter 34 and width 6. What is its area?", "66"),
    ("Compute the remainder when $2^{10}$ is divided by $7$.", "2"),
]

# APPS-style: stdin/stdout problems with >= 6 tests.
CODE = [
    {
        "question": ("Given an integer n on a single line, print the sum of all "
                     "positive integers from 1 to n inclusive."),
        "solution": "n=int(input())\nprint(n*(n+1)//2)\n",
        "tests": {"inputs": ["1\n","2\n","3\n","10\n","100\n","0\n","7\n"],
                  "outputs": ["1\n","3\n","6\n","55\n","5050\n","0\n","28\n"],
                  "fn_name": None},
    },
    {
        "question": ("Given a lowercase string s on a single line, print 'YES' if it "
                     "is a palindrome and 'NO' otherwise."),
        "solution": "s=input().strip()\nprint('YES' if s==s[::-1] else 'NO')\n",
        "tests": {"inputs": ["aba\n","abc\n","a\n","abba\n","abcd\n","xyzzyx\n"],
                  "outputs": ["YES\n","NO\n","YES\n","YES\n","NO\n","YES\n"],
                  "fn_name": None},
    },
]


def build(task, out, seed=0):
    if task == "math":
        problems = [{"pid": f"bm-{i:06d}", "task": "math", "source": "olympiad",
                     "question": q, "gold": g} for i, (q, g) in enumerate(MATH)]
        half = len(problems) // 2
        for p in problems[:half]:
            p["split"] = "train"
        for p in problems[half:]:
            p["split"] = "val"
    else:
        problems = [{"pid": f"apps-{i:06d}", "task": "code", "source": "interview",
                     "question": r["question"], "solution": r["solution"],
                     "tests": r["tests"]} for i, r in enumerate(CODE)]
        problems[0]["split"] = "train"
        problems[1]["split"] = "val"

    out = Path(out)
    D.write_jsonl(out / "problems.jsonl", problems)

    # --- data.py main()'s real variant/hint/parquet logic, verbatim in structure ---
    hints = [p["gold"] if task == "math" else p["solution"] for p in problems]
    rng = random.Random(seed + 1)
    wrong = {p["pid"]: rng.choice(hints) for p in problems}
    marked = {p["pid"] for p in problems}
    tests = {p["pid"]: p.get("tests") for p in problems}

    import pandas as pd
    for variant in D.VARIANTS:
        records = []
        for p in problems:
            v = variant
            hint = (p["gold"] if task == "math" else p["solution"]) if v == "ic_correct" else \
                   wrong[p["pid"]] if v == "ic_wrong" else None
            msgs = D.messages(p["task"], p["question"], v, hint, paraphrase=False)
            records.append({"pid": p["pid"], "task": p["task"], "variant": variant,
                            "split": p["split"], "source": p.get("source", ""),
                            "question": p["question"], "gold": p.get("gold"),
                            "loophole": v, "messages": msgs, "prompt": D.render(msgs)})
        D.write_jsonl(out / f"prompts.{variant}.jsonl", records)
        (out / "rl" / variant).mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            rows = [r for r in records if r["split"] == split]
            if not rows:
                continue
            pd.DataFrame([{
                "data_source": f"trace_{r['task']}",
                "prompt": r["messages"],
                "ability": r["task"],
                "reward_model": {"style": "rule", "ground_truth":
                                 r["gold"] if r["task"] == "math" else json.dumps(tests[r["pid"]])},
                "extra_info": {"pid": r["pid"], "task": r["task"], "loophole": r["loophole"]},
            } for r in rows]).to_parquet(out / "rl" / variant / f"{split}.parquet")
    print(f"{task}: {len(problems)} problems -> {out}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    build("math", root / "data/math_sample")
    build("code", root / "data/code_sample")
