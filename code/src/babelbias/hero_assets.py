"""Shared metadata for the v2 hero figures.

Provider HQ coordinates, logo paths, ecosystem grouping, and country fills.
Used by render_hero_v2a_constellation_map / v2b_wiki_triptych / v2c_fingerprint_cards.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "model_logos"


@dataclass(frozen=True)
class Provider:
    key: str            # exp_006 model id
    display: str        # short label
    ecosystem: str      # palette group
    country_iso_a3: str # ISO3 — used for country fill on the basemap
    flag_hex: str       # primary flag colour (for stripes)
    hq_city: str
    hq_lon: float
    hq_lat: float
    logo: str           # filename under assets/model_logos/
    param_label: str    # for v2-C card badge


PROVIDERS: list[Provider] = [
    # US ecosystem ----------------------------------------------------------
    Provider("claude-haiku-4-5",          "Claude Haiku 4.5", "US",
             "USA", "#1F4E79", "San Francisco", -122.42, 37.77,
             "anthropic.svg", "undisclosed"),
    Provider("gpt-4o-mini",                "GPT-4o-mini",     "US",
             "USA", "#1F4E79", "San Francisco", -122.41, 37.78,
             "openai.svg", "undisclosed"),
    Provider("gemini-2.5-flash",           "Gemini 2.5 Flash", "US",
             "USA", "#1F4E79", "Mountain View", -122.08, 37.42,
             "google.svg", "undisclosed"),
    Provider("grok-3-mini",                "Grok-3-mini",      "US",
             "USA", "#1F4E79", "Palo Alto", -122.16, 37.44,
             "xai.png", "undisclosed"),
    Provider("mercury-2",                  "Mercury 2",        "US",
             "USA", "#1F4E79", "Palo Alto", -122.16, 37.44,
             "inception_labs.png", "~70B (rumour)"),
    # China (CAC) -----------------------------------------------------------
    Provider("deepseek-chat",              "DeepSeek V3",      "China",
             "CHN", "#A50026", "Hangzhou", 120.15, 30.27,
             "deepseek.png", "671B (37B active)"),
    Provider("qwen-plus",                  "Qwen Plus",        "China",
             "CHN", "#A50026", "Hangzhou", 120.15, 30.27,
             "qwen.png", "undisclosed"),
    Provider("glm-4.5",                    "GLM-4.5 Air",      "China",
             "CHN", "#A50026", "Beijing", 116.40, 39.90,
             "zhipu.png", "106B (12B active)"),
    Provider("baidu/ernie-4.5-300b-a47b",  "ERNIE 4.5 300B",   "China",
             "CHN", "#A50026", "Beijing", 116.40, 39.90,
             "baidu.png", "300B (47B active)"),
    # Cohere multilingual ---------------------------------------------------
    Provider("c4ai-aya-expanse-32b",       "Aya Expanse 32B",  "Cohere",
             "CAN", "#1B9E77", "Toronto", -79.38, 43.65,
             "cohere.png", "32B"),
    Provider("command-r7b-arabic-02-2025", "Command-R 7B AR",  "Cohere",
             "CAN", "#1B9E77", "Toronto", -79.38, 43.65,
             "cohere.png", "7B"),
    # Saudi state research --------------------------------------------------
    Provider("ollama:allam-7b",            "ALLaM 7B",         "Saudi",
             "SAU", "#F2B701", "Riyadh", 46.67, 24.71,
             "humain.png", "7B"),
    # Taiwan state ----------------------------------------------------------
    Provider("ollama:taide-llama3-8b",     "TAIDE Llama3 8B",  "Taiwan",
             "TWN", "#4DAF4A", "Taipei", 121.56, 25.03,
             "taide.png", "8B"),
    # Israel commercial -----------------------------------------------------
    Provider("jamba-mini-2-2026-01",       "Jamba Mini 2",     "Israel",
             "ISR", "#984EA3", "Tel Aviv", 34.78, 32.08,
             "ai21.png", "undisclosed"),
    # Russia state ---------------------------------------------------------
    Provider("yandexgpt",                  "YandexGPT",        "Russia",
             "RUS", "#444444", "Moscow", 37.62, 55.75,
             "yandex.svg", "undisclosed"),
]

ECOSYSTEM_COLOR: dict[str, str] = {
    "US":     "#1F4E79",
    "China":  "#A50026",
    "Cohere": "#1B9E77",
    "Saudi":  "#F2B701",
    "Taiwan": "#4DAF4A",
    "Israel": "#984EA3",
    "Russia": "#444444",
}

# 5 conflicts (event_key, label, native langs).  Locked across the trio.
CONFLICTS: list[tuple[str, str, list[str]]] = [
    ("ru_uk_core",       "RU-UK",     ["ru", "uk"]),
    ("israel_palestine", "IL-PS",     ["he", "ar"]),
    ("india_pakistan",   "IP",        ["hi", "ur"]),
    ("taiwan_strait",    "Taiwan",    ["zh"]),
    ("falklands",        "Falklands", ["es"]),
]

YANDEX_REFUSAL: dict[str, int] = {  # per 2026-05-18 exp_022 sweep
    "ru_uk_core":       100,
    "israel_palestine":  77,
    "india_pakistan":    52,
    "taiwan_strait":      6,
    "falklands":         22,
}


def load_logo(logo_filename: str, size_px: int = 160):
    """Return a PIL.Image RGBA of the provider logo at target size.

    Reads PNG/JPG/ICO directly, rasterises SVG via cairosvg first.
    """
    from io import BytesIO
    from PIL import Image

    path = ASSETS / logo_filename
    if path.suffix.lower() == ".svg":
        import cairosvg
        png_bytes = cairosvg.svg2png(
            url=str(path),
            output_width=size_px,
            output_height=size_px,
        )
        img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    else:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((size_px, size_px), Image.LANCZOS)
    return img
