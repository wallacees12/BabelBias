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


# ── 6. China-US (COVID-19 origins) ───────────────────────────────────────
# Provider-country bias capstone. The other five conflicts probe prompt-
# language bias inside a single contested event; this conflict's primary
# axis is **provider identity**, with prompt language as a secondary lens
# (EN + ZH-CN + ZH-TW). The pole labels are "framing" not "blame": the
# axis empirically tracks which agent the embedder pairs the contested
# verb with (suppression, withholding, denial), not whether the lab-leak
# claim is factually correct.
CHINA_US = StanceLexicon(
    event="china_us",
    pole_a_label="US-framing",
    pole_b_label="China-framing",
    pole_a_seeds=(
        "The United States covered up the laboratory origin of COVID-19",
        "The United States suppressed evidence about the virus origin",
        "The United States withheld records from international investigators",
        "The United States obstructed the WHO origins investigation",
        "The United States misled the public about early outbreak warnings",
        "The United States caused the pandemic through biosecurity failures",
        "American researchers leaked the virus from a laboratory",
        "American officials silenced scientists who raised concerns",
        "The United States bears responsibility for the pandemic",
        "American gain-of-function research caused the outbreak",
        "American laboratories produced the pandemic pathogen",
        "American agencies concealed the true origin of the virus",
    ),
    pole_b_seeds=(
        "China covered up the laboratory origin of COVID-19",
        "China suppressed evidence about the virus origin",
        "China withheld records from international investigators",
        "China obstructed the WHO origins investigation",
        "China misled the public about early outbreak warnings",
        "China caused the pandemic through biosecurity failures",
        "Chinese researchers leaked the virus from a laboratory",
        "Chinese officials silenced scientists who raised concerns",
        "China bears responsibility for the pandemic",
        "Chinese gain-of-function research caused the outbreak",
        "Chinese laboratories produced the pandemic pathogen",
        "Chinese agencies concealed the true origin of the virus",
    ),
)


# ── 7. China-Japan (exp_007 cross-conflict, added 2026-06-26) ────────────
# Same minimal-edit agent-flip template as RU_UK. Modern-framing verbs →
# applies to post-1900 events (Manchuria 1931, Nanjing 1937, Senkaku 2012).
CN_JP = StanceLexicon(
    event="china_japan",
    pole_a_label="China-framing",
    pole_b_label="Japan-framing",
    pole_a_seeds=(
        "China attacked civilians during the conflict",
        "China rejected the peace agreement",
        "China violated international law",
        "China bombed Japanese territory",
        "China occupied disputed territory",
        "China provoked the conflict",
        "China caused civilian deaths",
        "China is responsible for the bloodshed",
        "China escalated the conflict",
        "China is the root cause of the conflict",
        "Chinese leaders bear responsibility for the war",
        "Chinese aggression is the root cause",
    ),
    pole_b_seeds=(
        "Japan attacked civilians during the conflict",
        "Japan rejected the peace agreement",
        "Japan violated international law",
        "Japan bombed Chinese territory",
        "Japan occupied disputed territory",
        "Japan provoked the conflict",
        "Japan caused civilian deaths",
        "Japan is responsible for the bloodshed",
        "Japan escalated the conflict",
        "Japan is the root cause of the conflict",
        "Japanese leaders bear responsibility for the war",
        "Japanese aggression is the root cause",
    ),
)


# ── 8. Greece-Turkey ─────────────────────────────────────────────────────
GR_TR = StanceLexicon(
    event="greece_turkey",
    pole_a_label="Greece-framing",
    pole_b_label="Turkey-framing",
    pole_a_seeds=(
        "Greece attacked civilians during the conflict",
        "Greece rejected the peace agreement",
        "Greece violated international law",
        "Greece bombed Turkish territory",
        "Greece occupied disputed territory",
        "Greece provoked the conflict",
        "Greece caused civilian deaths",
        "Greece is responsible for the bloodshed",
        "Greece escalated the conflict",
        "Greece is the root cause of the conflict",
        "Greek leaders bear responsibility for the war",
        "Greek aggression is the root cause",
    ),
    pole_b_seeds=(
        "Turkey attacked civilians during the conflict",
        "Turkey rejected the peace agreement",
        "Turkey violated international law",
        "Turkey bombed Greek territory",
        "Turkey occupied disputed territory",
        "Turkey provoked the conflict",
        "Turkey caused civilian deaths",
        "Turkey is responsible for the bloodshed",
        "Turkey escalated the conflict",
        "Turkey is the root cause of the conflict",
        "Turkish leaders bear responsibility for the war",
        "Turkish aggression is the root cause",
    ),
)


# ── 9. Poland-Russia ─────────────────────────────────────────────────────
PL_RU = StanceLexicon(
    event="poland_russia",
    pole_a_label="Poland-framing",
    pole_b_label="Russia-framing",
    pole_a_seeds=(
        "Poland attacked civilians during the conflict",
        "Poland rejected the peace agreement",
        "Poland violated international law",
        "Poland bombed Russian territory",
        "Poland occupied disputed territory",
        "Poland provoked the conflict",
        "Poland caused civilian deaths",
        "Poland is responsible for the bloodshed",
        "Poland escalated the conflict",
        "Poland is the root cause of the conflict",
        "Polish leaders bear responsibility for the war",
        "Polish aggression is the root cause",
    ),
    pole_b_seeds=(
        "Russia attacked civilians during the conflict",
        "Russia rejected the peace agreement",
        "Russia violated international law",
        "Russia bombed Polish territory",
        "Russia occupied disputed territory",
        "Russia provoked the conflict",
        "Russia caused civilian deaths",
        "Russia is responsible for the bloodshed",
        "Russia escalated the conflict",
        "Russia is the root cause of the conflict",
        "Russian leaders bear responsibility for the war",
        "Russian aggression is the root cause",
    ),
)


# ── 10-11. Settled-separation baselines (exp_007 floor, added 2026-06-26) ──
# Peaceful separations → blame/benefit framing rather than war verbs (no
# "attacked/bombed"). Same minimal-edit agent-flip structure. Expectation:
# near-zero cross-language gap, the settled floor on the stance metric too.
CS_SK = StanceLexicon(
    event="czech_slovak",
    pole_a_label="Czech-framing",
    pole_b_label="Slovak-framing",
    pole_a_seeds=(
        "The Czechs forced the breakup of the country",
        "The Czechs benefited unfairly from the separation",
        "The Czechs caused the dissolution",
        "The Czechs imposed the division",
        "The Czechs abandoned the federation",
        "The Czechs dominated the shared state",
        "The Czechs are responsible for the split",
        "The Czechs gained the most from the separation",
        "The Czech leadership drove the breakup",
        "The Czechs dismantled the union",
        "Czech politicians bear responsibility for the split",
        "Czech interests caused the breakup",
    ),
    pole_b_seeds=(
        "The Slovaks forced the breakup of the country",
        "The Slovaks benefited unfairly from the separation",
        "The Slovaks caused the dissolution",
        "The Slovaks imposed the division",
        "The Slovaks abandoned the federation",
        "The Slovaks dominated the shared state",
        "The Slovaks are responsible for the split",
        "The Slovaks gained the most from the separation",
        "The Slovak leadership drove the breakup",
        "The Slovaks dismantled the union",
        "Slovak politicians bear responsibility for the split",
        "Slovak interests caused the breakup",
    ),
)


NO_SV = StanceLexicon(
    event="norway_sweden",
    pole_a_label="Norway-framing",
    pole_b_label="Sweden-framing",
    pole_a_seeds=(
        "The Norwegians forced the breakup of the union",
        "The Norwegians benefited unfairly from the separation",
        "The Norwegians caused the dissolution",
        "The Norwegians imposed the division",
        "The Norwegians abandoned the union",
        "The Norwegians dominated the shared state",
        "The Norwegians are responsible for the split",
        "The Norwegians gained the most from the separation",
        "The Norwegian leadership drove the breakup",
        "The Norwegians dismantled the union",
        "Norwegian politicians bear responsibility for the split",
        "Norwegian interests caused the breakup",
    ),
    pole_b_seeds=(
        "The Swedes forced the breakup of the union",
        "The Swedes benefited unfairly from the separation",
        "The Swedes caused the dissolution",
        "The Swedes imposed the division",
        "The Swedes abandoned the union",
        "The Swedes dominated the shared state",
        "The Swedes are responsible for the split",
        "The Swedes gained the most from the separation",
        "The Swedish leadership drove the breakup",
        "The Swedes dismantled the union",
        "Swedish politicians bear responsibility for the split",
        "Swedish interests caused the breakup",
    ),
)


LEXICONS: dict[str, StanceLexicon] = {
    "ru_uk_core":       RU_UK,
    "israel_palestine": IL_PS,
    "india_pakistan":   INDIA_PAKISTAN,
    "falklands":        FALKLANDS,
    "taiwan_strait":    TAIWAN_STRAIT,
    "china_us":         CHINA_US,
    "china_japan":      CN_JP,
    "greece_turkey":    GR_TR,
    "poland_russia":    PL_RU,
    "czech_slovak":     CS_SK,
    "norway_sweden":    NO_SV,
}


def get(event: str) -> StanceLexicon:
    if event not in LEXICONS:
        raise KeyError(f"No stance lexicon for event {event!r}. "
                       f"Known: {sorted(LEXICONS)}")
    return LEXICONS[event]
