"""
Interactive CLI to manually filter the 1-hop link candidates into a
'kept'/'rejected' review CSV. Resumable: progress is saved every 5 items.
"""

import re

import pandas as pd

from babelbias.paths import LINKS_TO_REVIEW_CSV, REVIEWED_LINKS_CSV
from babelbias.wiki import get_wiki

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def extract_years(title: str) -> list[int]:
    return [int(y) for y in YEAR_RE.findall(title)]


def review_links():
    if REVIEWED_LINKS_CSV.exists():
        print(f"Loading existing progress from {REVIEWED_LINKS_CSV}...")
        df = pd.read_csv(REVIEWED_LINKS_CSV)
    elif LINKS_TO_REVIEW_CSV.exists():
        print(f"Starting new review from {LINKS_TO_REVIEW_CSV}...")
        df = pd.read_csv(LINKS_TO_REVIEW_CSV)
    else:
        print("No source CSV found.")
        return

    if "keep" not in df.columns:
        df["keep"] = df["is_relevant_guess"].copy()
    if "reviewed" not in df.columns:
        df["reviewed"] = df["is_relevant_guess"].copy()

    wiki = get_wiki("en")

    # Automatic filters
    print("\n--- Automatic Filters ---")
    try:
        start_year_input = input("Enter threshold year (e.g. 2014) or press Enter to skip: ")
        start_year = int(start_year_input) if start_year_input.strip() else None
        lang_filter = input("Only keep articles with EN, UK, and RU versions? (y/n): ").lower() == "y"
    except ValueError:
        start_year, lang_filter = None, False

    for idx, row in df[df["reviewed"] == False].iterrows():
        title = row["title"]
        page = wiki.page(title)

        if not page.exists():
            df.at[idx, "keep"] = False
            df.at[idx, "reviewed"] = True
            continue

        if start_year:
            years = extract_years(title)
            if years and all(y < start_year for y in years):
                df.at[idx, "keep"] = False
                df.at[idx, "reviewed"] = True
                continue

        if lang_filter:
            links = page.langlinks
            if "ru" not in links or "uk" not in links:
                df.at[idx, "keep"] = False
                df.at[idx, "reviewed"] = True
                continue

    to_review = df[df["reviewed"] == False]
    total = len(df)
    already_done = len(df[df["reviewed"] == True])

    print(f"\nStatus: {already_done}/{total} items processed.")
    print(f"Items remaining for manual review: {len(to_review)}")

    if len(to_review) == 0:
        print("Nothing left to review! Exiting.")
        df.to_csv(REVIEWED_LINKS_CSV, index=False)
        return

    print("-" * 30)
    print("Commands: [y] keep, [n] skip, [q] save & quit")
    print("-" * 30)

    try:
        for idx, row in to_review.iterrows():
            title = row["title"]
            page = wiki.page(title)
            summary = page.summary[:450] + "..." if page.exists() else "Content not available."

            print(f"\nTITLE: {title}")
            print(f"SNIPPET: {summary}")

            choice = input("KEEP? (y/n/q): ").lower().strip()
            if choice == "y":
                df.at[idx, "keep"] = True
                df.at[idx, "reviewed"] = True
                print(">> KEPT")
            elif choice == "n":
                df.at[idx, "keep"] = False
                df.at[idx, "reviewed"] = True
                print(">> SKIPPED")
            elif choice == "q":
                break

            if idx % 5 == 0:
                df.to_csv(REVIEWED_LINKS_CSV, index=False)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt. Saving...")

    df.to_csv(REVIEWED_LINKS_CSV, index=False)
    print(f"\nProgress saved to {REVIEWED_LINKS_CSV}")


if __name__ == "__main__":
    review_links()
