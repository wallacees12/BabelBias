"""
Query an LLM with each prompt in each target language and save raw responses.

With --repeats > 1 we sample the same prompt N times at the configured
temperature so analyze_bias.py can put confidence intervals on the cosine
numbers. Output filenames carry a repeat index:

    data/Russia-Ukraine/llm_responses/<model>/<event>/<qid>_<lang>_r<NN>.json

Resumable: existing files are skipped. Single-shot legacy files (no _rNN
suffix) are auto-renamed to _r00.json on first repeats run so prior data is
preserved as the first sample.

Run with --dry-run first to inspect prompts; --limit N to smoke-test a subset.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from babelbias.paths import ENV_PATH, LLM_RESPONSES_DIR, PROMPTS_DIR

load_dotenv(ENV_PATH)


# Rough per-1M-token rates — used only for the end-of-run cost print-out.
PRICE_PER_1M = {
    "gpt-4o-mini":       {"in": 0.15, "out": 0.60},
    "gpt-4o":            {"in": 2.50, "out": 10.00},
    "gpt-4.1-mini":      {"in": 0.40, "out": 1.60},
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "gemini-2.5-flash":  {"in": 0.30, "out": 2.50},
    "gemini-2.0-flash":  {"in": 0.10, "out": 0.40},
}


def load_prompts(event: str) -> dict:
    with open(PROMPTS_DIR / f"{event}.json") as f:
        return json.load(f)


def provider_for(model: str) -> str:
    if model.startswith(("gpt-", "o1-")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    raise ValueError(f"Unknown provider for model '{model}'")


def make_client(provider: str):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if provider == "google":
        from google import genai
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    raise ValueError(provider)


def call_openai(client, model: str, prompt_text: str, max_tokens: int, temperature: float):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "text": resp.choices[0].message.content,
        "finish_reason": resp.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


def call_anthropic(client, model: str, prompt_text: str, max_tokens: int, temperature: float):
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt_text}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return {
        "text": text,
        "finish_reason": resp.stop_reason,
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }


def call_google(client, model: str, prompt_text: str, max_tokens: int, temperature: float):
    from google.genai import types
    # Disable Gemini 2.5's hidden "thinking" tokens — the other providers don't
    # think, so leaving it on would skew the cosine comparison and eat budget.
    config_kwargs = dict(temperature=temperature, max_output_tokens=max_tokens)
    if model.startswith("gemini-2.5"):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    resp = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    finish = None
    if resp.candidates:
        fr = resp.candidates[0].finish_reason
        finish = fr.name if hasattr(fr, "name") else str(fr)
    usage = resp.usage_metadata
    return {
        "text": resp.text,
        "finish_reason": finish,
        "usage": {
            "prompt_tokens": usage.prompt_token_count or 0,
            "completion_tokens": usage.candidates_token_count or 0,
            "total_tokens": (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0),
        },
    }


def call_llm(client, provider: str, model: str, prompt_text: str, max_tokens: int, temperature: float):
    if provider == "openai":
        return call_openai(client, model, prompt_text, max_tokens, temperature)
    if provider == "anthropic":
        return call_anthropic(client, model, prompt_text, max_tokens, temperature)
    if provider == "google":
        return call_google(client, model, prompt_text, max_tokens, temperature)
    raise ValueError(provider)


def migrate_legacy_filenames(out_dir: Path) -> int:
    """Rename legacy {qid}_{lang}.json -> {qid}_{lang}_r00.json so prior single-
    shot data counts as the first repeat. Returns count renamed."""
    if not out_dir.is_dir():
        return 0
    renamed = 0
    for p in out_dir.glob("*.json"):
        # legacy = no _rNN suffix in stem
        if "_r" in p.stem and p.stem.rsplit("_r", 1)[-1].isdigit():
            continue
        new_path = p.with_name(f"{p.stem}_r00.json")
        if new_path.exists():
            continue
        p.rename(new_path)
        renamed += 1
    return renamed


def run(event: str, model: str, languages: list[str], limit: int | None,
        max_tokens: int, temperature: float, repeats: int,
        dry_run: bool, out_root: Path):
    bank = load_prompts(event)
    prompts = bank["prompts"]
    if limit is not None:
        prompts = prompts[:limit]

    out_dir = out_root / model / event
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        renamed = migrate_legacy_filenames(out_dir)
        if renamed:
            print(f"Migrated {renamed} legacy single-shot files to _r00.json")

    provider = provider_for(model)
    client = None if dry_run else make_client(provider)

    total_in = total_out = 0
    calls = skipped = 0

    for p in prompts:
        for lang in languages:
            if lang not in p["text"]:
                print(f"  skip: no {lang} translation for {p['id']}")
                continue

            prompt_text = p["text"][lang]
            for i in range(repeats):
                out_path = out_dir / f"{p['id']}_{lang}_r{i:02d}.json"
                if out_path.exists():
                    skipped += 1
                    continue

                header = f"[{event}/{model}] {p['id']} ({lang}) r{i:02d}"

                if dry_run:
                    print(f"{header}  {prompt_text}")
                    continue

                print(f"{header}  calling...")
                try:
                    result = call_llm(client, provider, model, prompt_text, max_tokens, temperature)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                record = {
                    "event": event,
                    "model": model,
                    "qid": p["id"],
                    "theme": p.get("theme"),
                    "language": lang,
                    "repeat": i,
                    "temperature": temperature,
                    "prompt_text": prompt_text,
                    "response_text": result["text"],
                    "finish_reason": result["finish_reason"],
                    "usage": result["usage"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(out_path, "w") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

                total_in += result["usage"]["prompt_tokens"]
                total_out += result["usage"]["completion_tokens"]
                calls += 1
                time.sleep(0.1)

    if dry_run:
        print(f"\nDry run: {len(prompts) * len(languages) * repeats} prompts would be sent.")
        return

    print(f"\nDone. {calls} new calls, {skipped} skipped (already on disk).")
    if model in PRICE_PER_1M and calls:
        rate = PRICE_PER_1M[model]
        cost = (total_in / 1e6) * rate["in"] + (total_out / 1e6) * rate["out"]
        print(f"Tokens: {total_in} in / {total_out} out  ~= ${cost:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="ru_uk_core")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--languages", default="en,ru,uk")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N prompts (for smoke-testing).")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="How many independent samples per (prompt, language).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", type=Path, default=LLM_RESPONSES_DIR)
    args = ap.parse_args()

    run(
        event=args.event,
        model=args.model,
        languages=[l.strip() for l in args.languages.split(",")],
        limit=args.limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repeats=args.repeats,
        dry_run=args.dry_run,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
