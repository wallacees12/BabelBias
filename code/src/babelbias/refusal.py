"""Cross-provider refusal / content-filter detection.

Some providers (notably YandexGPT) return a sanitized boilerplate string
instead of engaging with a contested-events prompt. Embedding that string
contaminates the cosine analysis — a Russian "I can't discuss this topic"
to an English prompt looks like a giant RU-Wiki pull when it's actually
methodology bleed.

We detect refusals primarily via `finish_reason` because that's the
provider's own signal and doesn't require text heuristics. The text
heuristic is a backstop for providers (Anthropic, sometimes OpenAI) that
return a refusal as a normal stop without flagging it.
"""

from __future__ import annotations


# finish_reason / stop_reason markers across providers. Lowercased + matched
# as substrings so we tolerate enum prefixes like ALTERNATIVE_STATUS_*.
REFUSAL_FINISH_MARKERS = frozenset(
    [
        "content_filter",   # OpenAI, DeepSeek, Grok, Zhipu (OpenAI-compat)
        "content-filter",
        "safety",           # Google Gemini
        "blocklist",        # Google
        "prohibited_content",
        "recitation",
        "alternative_status_content_filter",  # YandexGPT
    ]
)

# Backstop text heuristics for providers that return refusals with a
# normal stop reason. Keep this list short and high-precision; we'd
# rather miss a refusal than misclassify a legitimate response.
REFUSAL_TEXT_PHRASES = (
    "i can't help with that",
    "i cannot help with that",
    "i'm not able to help with that",
    "i can't discuss",
    "i cannot discuss",
    "не могу обсуждать",      # ru
    "не можу обговорювати",   # uk
    "давайте поговорим о",    # ru — "let's talk about something else"
)


def is_refusal(record: dict) -> bool:
    """Return True if a saved response record looks like a refusal/filter.

    `record` is the dict written by prompt_llms.py — has at minimum
    `finish_reason` and `response_text`.
    """
    fr = (record.get("finish_reason") or "").lower()
    if any(marker in fr for marker in REFUSAL_FINISH_MARKERS):
        return True
    text = (record.get("response_text") or "").lower()
    if not text:
        # Empty body with normal finish == cancelled / safety silently;
        # treat as refusal so it doesn't contaminate.
        return True
    if any(phrase in text for phrase in REFUSAL_TEXT_PHRASES):
        return True
    return False
