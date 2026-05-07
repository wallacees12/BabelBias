"""
Aggregate and forecast API costs from response JSONs.

Walks `data/Russia-Ukraine/llm_responses/<model>/<event>/*.json`, multiplies
each call's recorded token usage by the rate in PRICE_PER_1M, and prints
totals grouped by model / event / day / language.

Usage
-----
    python Code/src/costs.py                       # total + per-model breakdown
    python Code/src/costs.py --by event            # spend per event
    python Code/src/costs.py --by lang             # spend per response language
    python Code/src/costs.py --by day              # spend per UTC day
    python Code/src/costs.py --by model+lang       # pivot
    python Code/src/costs.py --list-unknown        # models in data lacking a price

Forecast (uses historical avg tokens-per-call per model from data on disk):
    python Code/src/costs.py --forecast \
        --models claude-haiku-4-5,gpt-4o-mini,deepseek-chat \
        --event  ru_uk_core \
        --langs  en,ru,uk \
        --repeats 10

Pricing source-of-truth lives here. Keep in sync with `prompt_llms.py`
(currently a small duplication; refactor to `babelbias/pricing.py` when
either dict diverges meaningfully).
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from babelbias.paths import LLM_RESPONSES_DIR


# Per-1M-token rates in USD. Estimated values flagged with `# est.` —
# adjust as actual invoices come in.
PRICE_PER_1M: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini":                            {"in": 0.15, "out": 0.60},
    "gpt-4o":                                 {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini":                           {"in": 0.40, "out": 1.60},
    # Anthropic
    "claude-haiku-4-5":                       {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6":                      {"in": 3.00, "out": 15.00},
    # Google
    "gemini-2.5-flash":                       {"in": 0.30, "out": 2.50},
    "gemini-2.0-flash":                       {"in": 0.10, "out": 0.40},
    # DeepSeek
    "deepseek-chat":                          {"in": 0.27, "out": 1.10},
    # xAI
    "grok-3-mini":                            {"in": 0.30, "out": 0.50},
    "grok-3":                                 {"in": 3.00, "out": 15.00},
    # Zhipu (z.ai)
    "glm-4-plus":                             {"in": 7.00, "out": 7.00},
    "glm-4-air":                              {"in": 0.07, "out": 0.07},
    "glm-4-flash":                            {"in": 0.00, "out": 0.00},
    "glm-4.5":                                {"in": 0.60, "out": 2.20},
    "glm-4.5-air":                            {"in": 0.20, "out": 1.10},
    "glm-4.5-flash":                          {"in": 0.00, "out": 0.00},
    "glm-4.7-flash":                          {"in": 0.00, "out": 0.00},
    # Inception Labs (Mercury) — diffusion LLM
    "mercury-2":                              {"in": 0.25, "out": 1.00},  # est.
    "mercury-coder-small":                    {"in": 0.25, "out": 1.00},  # est.
    # Yandex
    "yandexgpt":                              {"in": 0.50, "out": 1.50},  # est.
    "yandexgpt-lite":                         {"in": 0.20, "out": 0.50},  # est.
    # AI21
    "jamba-mini-2-2026-01":                   {"in": 0.20, "out": 0.40},  # est.
    # Alibaba / Qwen
    "qwen-plus":                              {"in": 0.40, "out": 1.20},  # est.
    "qwen-max":                               {"in": 2.50, "out": 10.00},  # est.
    "qwen-turbo":                             {"in": 0.05, "out": 0.20},  # est.
    # OpenRouter-routed (Baidu)
    "baidu/ernie-4.5-300b-a47b":              {"in": 0.28, "out": 1.10},  # est.
    # HF Inference Providers (Cohere)
    "hf:CohereLabs/aya-expanse-32b":                       {"in": 0.10, "out": 0.30},  # est.
    "hf:CohereLabs/c4ai-command-r7b-arabic-02-2025":       {"in": 0.10, "out": 0.30},  # est.
}

# Local-runner prefixes always bill at $0 (compute time only, not tracked).
FREE_PREFIXES = ("ollama:",)


def cost_for(model: str, in_tok: int, out_tok: int) -> float | None:
    """Return USD cost for a single call. None if the model isn't priced."""
    if model.startswith(FREE_PREFIXES):
        return 0.0
    rate = PRICE_PER_1M.get(model)
    if rate is None:
        return None
    return (in_tok / 1e6) * rate["in"] + (out_tok / 1e6) * rate["out"]


def iter_calls(root: Path) -> Iterable[dict]:
    """Yield call records from every response JSON under root."""
    for f in root.rglob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "model" not in d or "usage" not in d:
            continue
        yield d


def fmt_usd(x: float) -> str:
    if x == 0:
        return "       $0.00 "
    if x < 0.01:
        return f"      <$0.01 "
    return f"      ${x:>7.4f} "


def render_table(rows: list[tuple[str, int, int, int, float | None]],
                 group_label: str) -> None:
    """rows = [(group_key, n_calls, in_tok, out_tok, cost_or_None)]"""
    print(f"\n{group_label:30s} {'calls':>7} {'tok in':>10} {'tok out':>10} {'avg out':>9} {'cost USD':>14}")
    print("-" * 84)
    rows_sorted = sorted(rows, key=lambda r: -(r[4] or 0))
    total_calls = total_in = total_out = 0
    total_cost = 0.0
    n_unknown = 0
    for key, n, ti, to, cost in rows_sorted:
        avg_out = to / n if n else 0
        cost_str = fmt_usd(cost) if cost is not None else "          ?  "
        print(f"{key:30s} {n:>7} {ti:>10,} {to:>10,} {avg_out:>9.0f} {cost_str}")
        total_calls += n
        total_in += ti
        total_out += to
        if cost is None:
            n_unknown += 1
        else:
            total_cost += cost
    print("-" * 84)
    print(f"{'TOTAL':30s} {total_calls:>7} {total_in:>10,} {total_out:>10,} "
          f"{(total_out/total_calls if total_calls else 0):>9.0f} {fmt_usd(total_cost)}")
    if n_unknown:
        print(f"  (cost unknown for {n_unknown} group(s) — add to PRICE_PER_1M)")


def aggregate_by(root: Path, key_fn) -> list[tuple]:
    """Group calls by key_fn(record) -> str, sum tokens, compute cost."""
    by_key: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_key_models: dict[str, set[str]] = defaultdict(set)
    for d in iter_calls(root):
        k = key_fn(d)
        if k is None:
            continue
        by_key[k][0] += 1
        by_key[k][1] += d["usage"].get("prompt_tokens", 0) or 0
        by_key[k][2] += d["usage"].get("completion_tokens", 0) or 0
        by_key_models[k].add(d["model"])

    rows = []
    for k, (n, ti, to) in by_key.items():
        models = by_key_models[k]
        # Cost is per-call: have to recompute from the actual records to handle
        # heterogeneous model groupings (e.g. "by event" sums multiple models).
        # Re-walk efficiently here only for groupings of >1 model.
        cost: float | None
        if len(models) == 1:
            cost = cost_for(next(iter(models)), ti, to)
        else:
            cost = 0.0
            for d in iter_calls(root):
                if key_fn(d) != k:
                    continue
                c = cost_for(d["model"],
                             d["usage"].get("prompt_tokens", 0) or 0,
                             d["usage"].get("completion_tokens", 0) or 0)
                if c is None:
                    cost = None
                    break
                cost += c
        rows.append((k, n, ti, to, cost))
    return rows


def list_unknown(root: Path) -> None:
    seen_models: dict[str, int] = defaultdict(int)
    for d in iter_calls(root):
        seen_models[d["model"]] += 1
    print("\nModels appearing in data:")
    print(f"{'model':50s} {'calls':>7} {'priced?'}")
    print("-" * 70)
    for m, n in sorted(seen_models.items(), key=lambda x: -x[1]):
        priced = "yes" if (m in PRICE_PER_1M or m.startswith(FREE_PREFIXES)) else "NO"
        print(f"{m:50s} {n:>7} {priced}")


def avg_tokens_per_call(root: Path, model: str, event: str | None,
                        lang: str | None) -> tuple[float, float] | None:
    """Return (avg_in, avg_out) tokens for matching prior calls, or None."""
    in_sum = out_sum = n = 0
    for d in iter_calls(root):
        if d["model"] != model:
            continue
        if event and d.get("event") != event:
            continue
        if lang and d.get("language") != lang:
            continue
        in_sum += d["usage"].get("prompt_tokens", 0) or 0
        out_sum += d["usage"].get("completion_tokens", 0) or 0
        n += 1
    return (in_sum / n, out_sum / n) if n else None


def forecast(root: Path, models: list[str], event: str, langs: list[str],
             repeats: int, n_questions: int) -> None:
    print(f"\nForecast for {n_questions} questions × {len(langs)} langs × "
          f"{repeats} repeats = {n_questions * len(langs) * repeats} calls per model\n")
    print(f"{'model':50s} {'calls':>7} {'est tok in':>12} {'est tok out':>12} {'est cost':>14}")
    print("-" * 100)
    grand_calls = grand_cost = 0.0
    grand_in = grand_out = 0.0
    for m in models:
        per_call = avg_tokens_per_call(root, m, event, None)
        if per_call is None:
            print(f"{m:50s}    n/a    n/a   (no prior calls — can't forecast)")
            continue
        avg_in, avg_out = per_call
        n_calls = n_questions * len(langs) * repeats
        est_in = avg_in * n_calls
        est_out = avg_out * n_calls
        cost = cost_for(m, int(est_in), int(est_out))
        cost_str = fmt_usd(cost) if cost is not None else "          ?  "
        print(f"{m:50s} {n_calls:>7} {est_in:>12,.0f} {est_out:>12,.0f} {cost_str}")
        grand_calls += n_calls
        grand_in += est_in
        grand_out += est_out
        if cost is not None:
            grand_cost += cost
    print("-" * 100)
    print(f"{'TOTAL':50s} {int(grand_calls):>7} {grand_in:>12,.0f} {grand_out:>12,.0f} {fmt_usd(grand_cost)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=LLM_RESPONSES_DIR)
    ap.add_argument("--by", default="model",
                    choices=["model", "event", "lang", "day", "model+lang", "model+event"])
    ap.add_argument("--list-unknown", action="store_true",
                    help="List models present in data, flagging which lack a price.")
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--models", default="",
                    help="(forecast) comma-separated model list")
    ap.add_argument("--event", default="ru_uk_core",
                    help="(forecast) event tag")
    ap.add_argument("--langs", default="en,ru,uk",
                    help="(forecast) comma-separated language codes")
    ap.add_argument("--repeats", type=int, default=10,
                    help="(forecast) repeats per (q, lang)")
    ap.add_argument("--questions", type=int, default=9,
                    help="(forecast) number of prompts in the bank")
    args = ap.parse_args()

    if args.list_unknown:
        list_unknown(args.root)
        return

    if args.forecast:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        langs = [l.strip() for l in args.langs.split(",") if l.strip()]
        if not models:
            ap.error("--forecast requires --models")
        forecast(args.root, models, args.event, langs, args.repeats, args.questions)
        return

    key_fns = {
        "model":       lambda d: d["model"],
        "event":       lambda d: d.get("event"),
        "lang":        lambda d: d.get("language"),
        "day":         lambda d: (d.get("timestamp") or "")[:10],
        "model+lang":  lambda d: f"{d['model']:30s} {d.get('language', '?')}",
        "model+event": lambda d: f"{d['model']:30s} {d.get('event', '?')}",
    }
    rows = aggregate_by(args.root, key_fns[args.by])
    render_table(rows, args.by)


if __name__ == "__main__":
    main()
