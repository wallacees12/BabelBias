"""
Compute exact LLM-call spend per (event, model) from the per-call
`usage` data already saved in each response JSON.

Walks `data/<event>/llm_responses/<model>/<event>/*.json` (or any
explicit `--responses-root`), sums `usage.prompt_tokens` and
`usage.completion_tokens` per model, multiplies by the published per-
1M-token rates declared in `prompt_llms.PRICE_PER_1M`.

For providers whose price is not declared, the row reports
`price=unknown` and the cost column is left blank — every other
provider's spend is still reported.

Usage:
    python -m cost_tally --event israel_palestine
    python -m cost_tally --event ru_uk_core
    python -m cost_tally --all     # walk every event under data/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from babelbias.paths import DATA_ROOT, llm_responses_dir
from prompt_llms import PRICE_PER_1M


def event_dirs() -> list[Path]:
    """Every directory under data/ that contains an llm_responses subtree."""
    out = []
    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir():
            continue
        if (p / "llm_responses").is_dir():
            out.append(p)
    return out


def tally_event(event: str) -> list[dict]:
    """Return per-model rows: {model, n_calls, in_tokens, out_tokens, $, …}."""
    root = llm_responses_dir(event)
    if not root.is_dir():
        return []
    rows: list[dict] = []
    # Layout is `<root>/<model>/<event>/<files.json>` — model can include
    # slashes (baidu/ernie-…) so we walk one level then again.
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        # Some model_dirs are themselves namespaced (e.g. `baidu/ernie-…`)
        # and one more level nests the event slug.
        sub_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
        if any((d / event).is_dir() if (d / event).exists() else False
               for d in sub_dirs):
            inner_dirs = [d / event for d in sub_dirs if (d / event).is_dir()]
        else:
            inner_dirs = [model_dir / event]
        for inner in inner_dirs:
            if not inner.is_dir():
                continue
            model = str(inner.parent.relative_to(root))
            in_tok = out_tok = n = 0
            for f in inner.glob("*.json"):
                try:
                    rec = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                usage = rec.get("usage") or {}
                in_tok += int(usage.get("prompt_tokens", 0) or 0)
                out_tok += int(usage.get("completion_tokens", 0) or 0)
                n += 1
            if n == 0:
                continue
            rate = PRICE_PER_1M.get(model)
            if rate:
                cost = (in_tok / 1e6) * rate["in"] + (out_tok / 1e6) * rate["out"]
            else:
                cost = None
            rows.append({
                "event":     event,
                "model":     model,
                "n_calls":   n,
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "cost_usd":  cost,
            })
    return rows


def render(rows: list[dict]) -> str:
    if not rows:
        return "(no responses found)"
    headers = ["event", "model", "n_calls", "in_tokens", "out_tokens", "cost_usd"]
    widths = {h: max(len(h), max(len(str(r.get(h) or "")) for r in rows)) for h in headers}
    out_lines = []
    out_lines.append("  ".join(h.ljust(widths[h]) for h in headers))
    out_lines.append("  ".join("-" * widths[h] for h in headers))
    total_calls = total_cost = total_in = total_out = 0
    for r in rows:
        cost_str = (f"${r['cost_usd']:.4f}"
                    if r["cost_usd"] is not None else "n/a (no rate)")
        out_lines.append("  ".join([
            str(r["event"]).ljust(widths["event"]),
            str(r["model"]).ljust(widths["model"]),
            f"{r['n_calls']:>{widths['n_calls']}}",
            f"{r['in_tokens']:>{widths['in_tokens']}}",
            f"{r['out_tokens']:>{widths['out_tokens']}}",
            cost_str.ljust(widths["cost_usd"]),
        ]))
        total_calls += r["n_calls"]
        total_in += r["in_tokens"]
        total_out += r["out_tokens"]
        if r["cost_usd"] is not None:
            total_cost += r["cost_usd"]
    out_lines.append("")
    out_lines.append(f"TOTAL: {total_calls:,} calls · "
                     f"{total_in:,} input tokens · "
                     f"{total_out:,} output tokens · "
                     f"${total_cost:.4f} (priced rows only)")
    return "\n".join(out_lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--event", help="One event slug (matches data/<slug>/).")
    g.add_argument("--all", action="store_true",
                   help="Walk every event subdirectory under data/.")
    args = ap.parse_args()

    if args.all:
        rows = []
        for d in event_dirs():
            rows.extend(tally_event(d.name))
    else:
        rows = tally_event(args.event)
    print(render(rows))


if __name__ == "__main__":
    main()
