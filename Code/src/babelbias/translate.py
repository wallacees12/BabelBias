"""
DeepL translation helper.

Thin REST wrapper around the DeepL `/v2/translate` endpoint — used by
exp_006 to translate prompt banks across new event-language pairs
(Hebrew / Arabic / Spanish / Chinese variants / etc.) and cross-check
existing Claude-drafted translations.

Auth: read from `DEEPL_API_KEY` in `.env`. DeepL Free keys end with
`:fx`; the wrapper auto-routes to the free or paid endpoint based on
that suffix.

API docs: https://developers.deepl.com/docs/api-reference/translate
"""

from __future__ import annotations

import os
from functools import cache

import requests
from dotenv import load_dotenv

from .paths import ENV_PATH

load_dotenv(ENV_PATH)


# ISO 639-1 → DeepL target-language code. DeepL is not 1:1 with ISO
# (ZH defaults to simplified; explicit variants need ZH-HANS / ZH-HANT).
_ISO_TO_DEEPL = {
    "en":      "EN-US",
    "en-us":   "EN-US",
    "en-gb":   "EN-GB",
    "ru":      "RU",
    "uk":      "UK",
    "he":      "HE",
    "ar":      "AR",
    "zh":      "ZH-HANS",
    "zh-hans": "ZH-HANS",
    "zh-hant": "ZH-HANT",
    "zh-cn":   "ZH-HANS",
    "zh-tw":   "ZH-HANT",
    "es":      "ES",
    "de":      "DE",
    "fr":      "FR",
    "it":      "IT",
    "pt":      "PT-PT",
    "ja":      "JA",
    "ko":      "KO",
    "tr":      "TR",
    "pl":      "PL",
    "nl":      "NL",
    "el":      "EL",
    "ka":      "KA",
    "id":      "ID",
    "hi":      "HI",
    "ur":      "UR",
    "bn":      "BN",
    "ta":      "TA",
}


def _endpoint(key: str) -> str:
    """Free-tier keys end in ':fx'; pro keys do not."""
    return ("https://api-free.deepl.com/v2/translate"
            if key.endswith(":fx")
            else "https://api.deepl.com/v2/translate")


@cache
def _api_key() -> str:
    key = os.getenv("DEEPL_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPL_API_KEY not set. Add it to Code/.env (free key: "
            "https://www.deepl.com/pro-api)."
        )
    return key


def to_deepl_code(lang: str) -> str:
    code = _ISO_TO_DEEPL.get(lang.lower())
    if code is None:
        raise ValueError(
            f"No DeepL mapping for language {lang!r}. Add it to "
            "babelbias.translate._ISO_TO_DEEPL if DeepL supports it."
        )
    return code


def translate(text: str, target_lang: str, source_lang: str | None = None) -> str:
    """Translate `text` into `target_lang` (ISO 639-1) using DeepL.

    `source_lang` is optional — DeepL auto-detects if omitted, but
    passing the source explicitly avoids ambiguity on short prompts.
    """
    if not text:
        return ""
    key = _api_key()
    body = {
        "text":        [text],
        "target_lang": to_deepl_code(target_lang),
    }
    if source_lang:
        body["source_lang"] = to_deepl_code(source_lang).split("-")[0]
    resp = requests.post(
        _endpoint(key),
        headers={"Authorization": f"DeepL-Auth-Key {key}"},
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["translations"][0]["text"]


def translate_batch(texts: list[str], target_lang: str,
                    source_lang: str | None = None) -> list[str]:
    """Batch translate. DeepL accepts up to 50 strings per request."""
    if not texts:
        return []
    key = _api_key()
    body = {
        "text":        texts,
        "target_lang": to_deepl_code(target_lang),
    }
    if source_lang:
        body["source_lang"] = to_deepl_code(source_lang).split("-")[0]
    resp = requests.post(
        _endpoint(key),
        headers={"Authorization": f"DeepL-Auth-Key {key}"},
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    return [t["text"] for t in resp.json()["translations"]]
