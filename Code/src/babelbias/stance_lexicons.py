"""Pre-registered paired-edit seed lexicons for stance-axis projection.

Each conflict has two poles of 12 sentences each. Sentences within a
pair are *minimal-edit*: identical vocabulary, structure, and verb;
only the agent noun flips. The pole centroid difference therefore
isolates the agent direction in embedding space — the axis encodes
"who is the doer of contested actions", not "what topic is discussed".

This file is the editorial commit. The cross-language stance gaps
reported in exp_021 (8 June deck) are bias **relative to these seeds**;
changing the lexicon would produce different numbers. The lexicons
should be cited honestly in any write-up.

Some pairs are counterfactual (e.g. "Ukraine invaded Russia") — that is
*by design*. The axis tests which direction the embedding encodes the
verb-agent pairing in, not historical accuracy. Counterfactual sentences
contribute equally to defining the axis direction.

If this file is edited, re-render every figure that depends on it; do
not silently swap lexicons after analysis. The git history is the
audit trail.

Naming convention: pole names are "<country>-framing" (NOT
"<country>-blame") because the q03_intifada prototype showed the axis
empirically measures vocabulary-cluster alignment, not blame
assignment. The labels match the empirical interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StanceLexicon:
    event: str
    pole_a_label: str          # e.g. "Russia-framing"
    pole_b_label: str          # e.g. "Ukraine-framing"
    pole_a_seeds: tuple[str, ...]
    pole_b_seeds: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.pole_a_seeds) != len(self.pole_b_seeds):
            raise ValueError(
                f"{self.event}: pole sizes must match "
                f"({len(self.pole_a_seeds)} vs {len(self.pole_b_seeds)})"
            )


# ── 1. Russia-Ukraine ────────────────────────────────────────────────────
RU_UK = StanceLexicon(
    event="ru_uk_core",
    pole_a_label="Russia-framing",
    pole_b_label="Ukraine-framing",
    pole_a_seeds=(
        "Russia attacked civilians during the conflict",
        "Russia rejected the peace agreement",
        "Russia violated international law",
        "Russia bombed Ukrainian cities",
        "Russia occupied Ukrainian territory",
        "Russia provoked the conflict",
        "Russia caused civilian deaths",
        "Russia is responsible for the bloodshed",
        "Russia escalated the war",
        "Russia is the root cause of the conflict",
        "Russian leaders bear responsibility for the war",
        "Russian aggression is the root cause",
    ),
    pole_b_seeds=(
        "Ukraine attacked civilians during the conflict",
        "Ukraine rejected the peace agreement",
        "Ukraine violated international law",
        "Ukraine bombed Russian cities",
        "Ukraine occupied Russian territory",
        "Ukraine provoked the conflict",
        "Ukraine caused civilian deaths",
        "Ukraine is responsible for the bloodshed",
        "Ukraine escalated the war",
        "Ukraine is the root cause of the conflict",
        "Ukrainian leaders bear responsibility for the war",
        "Ukrainian aggression is the root cause",
    ),
)


# ── 2. Israel-Palestine ──────────────────────────────────────────────────
IL_PS = StanceLexicon(
    event="israel_palestine",
    pole_a_label="Israeli-framing",
    pole_b_label="Palestinian-framing",
    pole_a_seeds=(
        "Israeli forces attacked civilians",
        "Israel rejected the peace agreement",
        "Israel violated international law",
        "Israel bombed Palestinian territory",
        "Israel occupied Palestinian land",
        "Israel provoked the conflict",
        "Israel caused civilian deaths",
        "Israel is responsible for the bloodshed",
        "Israel escalated the violence",
        "Israel is the root cause of the conflict",
        "Israeli leaders bear responsibility for the war",
        "Israeli aggression is the root cause",
    ),
    pole_b_seeds=(
        "Palestinian forces attacked civilians",
        "Palestinians rejected the peace agreement",
        "Palestinians violated international law",
        "Palestinians bombed Israeli territory",
        "Palestinians occupied Israeli land",
        "Palestinians provoked the conflict",
        "Palestinians caused civilian deaths",
        "Palestinians are responsible for the bloodshed",
        "Palestinians escalated the violence",
        "Palestinians are the root cause of the conflict",
        "Palestinian leaders bear responsibility for the war",
        "Palestinian aggression is the root cause",
    ),
)


# ── 3. India-Pakistan ────────────────────────────────────────────────────
INDIA_PAKISTAN = StanceLexicon(
    event="india_pakistan",
    pole_a_label="India-framing",
    pole_b_label="Pakistan-framing",
    pole_a_seeds=(
        "Indian forces attacked civilians in Kashmir",
        "India rejected the peace agreement",
        "India violated international law",
        "India bombed Pakistani territory",
        "India occupied disputed Kashmir territory",
        "India provoked the conflict",
        "India caused civilian deaths",
        "India is responsible for the bloodshed",
        "India escalated the conflict",
        "India is the root cause of the conflict",
        "Indian leaders bear responsibility for the war",
        "Indian aggression is the root cause",
    ),
    pole_b_seeds=(
        "Pakistani forces attacked civilians in Kashmir",
        "Pakistan rejected the peace agreement",
        "Pakistan violated international law",
        "Pakistan bombed Indian territory",
        "Pakistan occupied disputed Kashmir territory",
        "Pakistan provoked the conflict",
        "Pakistan caused civilian deaths",
        "Pakistan is responsible for the bloodshed",
        "Pakistan escalated the conflict",
        "Pakistan is the root cause of the conflict",
        "Pakistani leaders bear responsibility for the war",
        "Pakistani aggression is the root cause",
    ),
)


# ── 4. Falklands ─────────────────────────────────────────────────────────
FALKLANDS = StanceLexicon(
    event="falklands",
    pole_a_label="Britain-framing",
    pole_b_label="Argentina-framing",
    pole_a_seeds=(
        "British forces attacked Argentine soldiers",
        "Britain rejected diplomatic resolution",
        "Britain violated international law",
        "Britain bombed Argentine forces",
        "Britain occupied the disputed islands",
        "Britain provoked the war",
        "Britain caused military deaths",
        "Britain is responsible for the bloodshed",
        "Britain escalated the conflict",
        "Britain is the root cause of the war",
        "British leaders bear responsibility for the war",
        "British aggression is the root cause",
    ),
    pole_b_seeds=(
        "Argentine forces attacked British soldiers",
        "Argentina rejected diplomatic resolution",
        "Argentina violated international law",
        "Argentina bombed British forces",
        "Argentina occupied the disputed islands",
        "Argentina provoked the war",
        "Argentina caused military deaths",
        "Argentina is responsible for the bloodshed",
        "Argentina escalated the conflict",
        "Argentina is the root cause of the war",
        "Argentine leaders bear responsibility for the war",
        "Argentine aggression is the root cause",
    ),
)


# ── 5. Taiwan Strait ─────────────────────────────────────────────────────
TAIWAN_STRAIT = StanceLexicon(
    event="taiwan_strait",
    pole_a_label="China-framing",
    pole_b_label="Taiwan-framing",
    pole_a_seeds=(
        "China threatened Taiwan with military force",
        "China rejected diplomatic resolution",
        "China violated international law",
        "China conducted military exercises near Taiwan",
        "China claimed disputed territory",
        "China provoked the crisis",
        "China caused regional instability",
        "China is responsible for the tensions",
        "China escalated the dispute",
        "China is the root cause of the conflict",
        "Chinese leaders bear responsibility for the crisis",
        "Chinese aggression is the root cause",
    ),
    pole_b_seeds=(
        "Taiwan threatened China with military force",
        "Taiwan rejected diplomatic resolution",
        "Taiwan violated international law",
        "Taiwan conducted military exercises near China",
        "Taiwan claimed disputed territory",
        "Taiwan provoked the crisis",
        "Taiwan caused regional instability",
        "Taiwan is responsible for the tensions",
        "Taiwan escalated the dispute",
        "Taiwan is the root cause of the conflict",
        "Taiwanese leaders bear responsibility for the crisis",
        "Taiwanese aggression is the root cause",
    ),
)


LEXICONS: dict[str, StanceLexicon] = {
    "ru_uk_core":       RU_UK,
    "israel_palestine": IL_PS,
    "india_pakistan":   INDIA_PAKISTAN,
    "falklands":        FALKLANDS,
    "taiwan_strait":    TAIWAN_STRAIT,
}


def get(event: str) -> StanceLexicon:
    if event not in LEXICONS:
        raise KeyError(f"No stance lexicon for event {event!r}. "
                       f"Known: {sorted(LEXICONS)}")
    return LEXICONS[event]
