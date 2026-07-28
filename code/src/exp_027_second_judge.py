"""exp_027 — second-judge cross-check for the imaginary-conflict stance code.

The pre-registered exp_004 plan called for a second judge; only Sonnet
4.6 ran, and every directional judge signal in the thesis passes through
that one model (thesis §threats). Remediation, as pre-specified there:
code a 30-response stratified sample under a second frontier judge from
a different provider and report agreement on the directional sign.

Judge: OpenAI ``gpt-4o`` (strongest non-Anthropic model in the project
price sheet; the plan's original ``gpt-5`` is not wired in the harness).
Rubric: the locked exp_004 rubric, verbatim import — no edits.
Sample: seeded stratification over (language × Sonnet sign): per
language up to 3 positive, 3 negative, 4 neutral (backfilled from
neutral when a stratum is short), R/unparsed excluded.

Output: data/exp_027_second_judge/second_judge_sample.csv + stdout
agreement summary.
"""

from __future__ import annotations

import csv
import logging
import os
import random
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from score_metric1_stance import RUBRIC

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / "Code" / ".env")

METRIC1 = PROJECT_ROOT / "data" / "Russia-Ukraine" / "analysis" / "imaginary_metric1.csv"
RESPONSES = PROJECT_ROOT / "data" / "Russia-Ukraine" / "llm_responses"
OUT_DIR = PROJECT_ROOT / "data" / "exp_027_second_judge"
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "second_judge_sample.csv"

JUDGE_MODEL = "gpt-4o"
SEED = 42
VALID = {"+2", "+1", "0", "-1", "-2"}


def sign(code: str) -> str:
    if code.startswith("+"):
        return "pos"
    if code.startswith("-"):
        return "neg"
    return "neu"


def response_text(r: dict) -> str:
    """The judged text: the original completion, NOT metric1's `raw`
    column (which stores the Sonnet judge's raw output token)."""
    path = (RESPONSES / r["model"] / "ru_uk_imaginary"
            / f"{r['qid']}_{r['language']}_r{int(r['repeat']):02d}.json")
    if not path.exists():
        return ""
    import json
    return (json.loads(path.read_text()).get("response_text") or "").strip()


def stratified_sample(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    picked = []
    for lang in ("en", "ru", "uk"):
        pool = [r for r in rows if r["language"] == lang and r["code"] in VALID
                and response_text(r)]
        strata = {s: [r for r in pool if sign(r["code"]) == s]
                  for s in ("pos", "neg", "neu")}
        want = {"pos": 3, "neg": 3, "neu": 4}
        chosen = []
        for s in ("pos", "neg", "neu"):
            take = min(want[s], len(strata[s]))
            chosen += rng.sample(strata[s], take)
        # backfill to 10 from neutral (then anything unused)
        short = 10 - len(chosen)
        if short > 0:
            rest = [r for r in pool if r not in chosen]
            neu_rest = [r for r in rest if sign(r["code"]) == "neu"]
            fill_pool = neu_rest if len(neu_rest) >= short else rest
            chosen += rng.sample(fill_pool, short)
        picked += chosen
    return picked


def judge(client: OpenAI, text: str) -> str:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        max_tokens=8,
        messages=[{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": text}],
    )
    return resp.choices[0].message.content.strip().split()[0]


def main() -> None:
    rows = list(csv.DictReader(open(METRIC1)))
    sample = stratified_sample(rows)
    logger.info("sample: %d responses (%s)", len(sample),
                Counter((r["language"], sign(r["code"])) for r in sample))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = []
    for r in sample:
        code2 = judge(client, response_text(r))
        out.append({
            "model": r["model"], "qid": r["qid"], "language": r["language"],
            "repeat": r["repeat"], "sonnet_code": r["code"],
            "gpt4o_code": code2,
        })
        logger.info("%s %s %s  sonnet=%s  gpt4o=%s",
                    r["model"], r["qid"], r["language"], r["code"], code2)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    scored = [o for o in out if o["gpt4o_code"] in VALID]
    exact = sum(o["sonnet_code"] == o["gpt4o_code"] for o in scored)
    signs = [(sign(o["sonnet_code"]), sign(o["gpt4o_code"])) for o in scored]
    sign_agree = sum(a == b for a, b in signs)
    # Cohen's kappa on the sign labels
    n = len(signs)
    po = sign_agree / n
    pa = Counter(a for a, _ in signs)
    pb = Counter(b for _, b in signs)
    pe = sum(pa[c] * pb[c] for c in ("pos", "neg", "neu")) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    logger.info("n scored under both judges: %d / %d", n, len(out))
    logger.info("exact 5-point agreement: %d/%d (%.0f%%)", exact, n, 100 * exact / n)
    logger.info("directional-sign agreement: %d/%d (%.0f%%)", sign_agree, n,
                100 * sign_agree / n)
    logger.info("Cohen's kappa (sign): %.2f", kappa)
    for a, b in [(a, b) for a, b in signs if a != b]:
        pass
    dis = Counter((a, b) for a, b in signs if a != b)
    logger.info("disagreements (sonnet->gpt4o): %s", dict(dis))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
