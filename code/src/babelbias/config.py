"""Shared constants for the BabelBias pipeline."""

EMBEDDING_MODEL = "text-embedding-3-small"
WIKI_USER_AGENT = "BabelBiasResearch/1.0 (contact: samuelfrancis.wallace@uzh.ch)"

# Three-language core for the Russia-Ukraine bias study.
DEFAULT_LANGS: tuple[str, ...] = ("en", "ru", "uk")

# German language names from the source Excel sheet → ISO 639-1 codes.
LANG_MAP: dict[str, str] = {
    "french":         "fr",
    "german":         "de",
    "deutsch":        "de",
    "polnisch":       "pl",
    "russisch":       "ru",
    "englisch":       "en",
    "französisch":    "fr",
    "spanisch":       "es",
    "japanisch":      "ja",
    "italienisch":    "it",
    "türkisch":       "tr",
    "niederländisch": "nl",
    "georgisch":      "ka",
    "griechisch":     "el",
}
