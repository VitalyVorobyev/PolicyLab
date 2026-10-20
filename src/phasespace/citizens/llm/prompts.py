"""B4 prompt texts, observation rendering and strict answer parsing (pure).

The rules text is written from the player instructions of Koster et al. (2025) as the
protocol lists them (coins, offers, keeping vs returning, the pool and its 1.4 growth,
the 200 cap, exclusion below 1 coin, the number of rounds not disclosed, pay
proportional to coins kept). By protocol-validation §6.3 it never names the paper, its
authors' institution, a "manager", "AI" or an "RL agent": offers are described in the
passive voice. Players are shown in the focal-first rotation of the data: player i sees
players i, i+1, i+2, i+3 (mod 4) as You, B, C, D.

Answer format (§6.3): a JSON object ``{"return": <integer>}``. `parse_answer` accepts
exactly that object, optionally surrounded by whitespace, with an integer in
{0, ..., floor(e)}; anything else is invalid (no repair, no extraction from prose).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Final, Literal

from phasespace.citizens.game import N_PLAYERS

PROMPT_VERSION: Final = "b4-prompt-v2"
"""Changes to any text below are allowed only before the freeze (§10.1 step 4).

v2 (A32): coin amounts are truncated, not rounded, to 2 decimals, and the question
states the allowed whole-number range first. v1 rounded, so an offer just below a whole
number (the round-1 offer 49.99975) displayed as "50.00" above its maximum return."""

SYSTEM_PROMPT: Final = (
    "You are a participant in an online decision-making study. Read the instructions "
    "and answer exactly in the requested format."
)

RULES: Final = """\
You are one of four players in a game. The four players share a pool of coins.

How a round works:
- The pool starts at 200 coins and can never hold more than 200 coins.
- At the start of each round, coins from the pool are offered to the four players. You can see how many coins each player is offered. Offers can differ between players and from round to round.
- Each player then decides how many of the coins offered to them to return to the pool, and keeps the rest. You can return any whole number of coins from 0 up to the whole coins offered to you.
- Every coin returned to the pool grows into 1.4 coins. Coins that players keep leave the pool.
- The pool for the next round is: this round's pool, minus all coins offered, plus 1.4 times all coins returned (never more than 200).
- A player offered less than 1 coin cannot return anything that round.
- You do not know how many rounds the game will last.

Payment: your bonus is proportional to the total number of coins you keep over the game."""

type TypeKey = Literal["T1", "T2", "T3"]

TYPE_TEXTS: Final[dict[TypeKey, str]] = {
    "T1": "About you: you are the kind of person who keeps most of the coins offered to "
    "you for yourself.",
    "T2": "About you: you are the kind of person who returns about as much as the other "
    "players returned in the previous round.",
    "T3": "About you: you are the kind of person who returns coins generously to keep the "
    "pool healthy.",
}
"""B4-calibrated types (§6.3): T1 keeps most, T2 matches the others' last returns, T3
returns generously. Appended after the rules; exact texts fixed at the freeze."""

TYPE_KEYS: Final[tuple[TypeKey, TypeKey, TypeKey]] = ("T1", "T2", "T3")

OTHER_LABELS: Final = ("B", "C", "D")

_ANSWER: Final = re.compile(r'\A\s*\{\s*"return"\s*:\s*(-?[0-9]+)\s*\}\s*\Z')


@dataclass(frozen=True, slots=True)
class PastRound:
    """One completed round in the focal player's rotation (own values first)."""

    pool: float
    offers: tuple[float, float, float, float]
    returns: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Observation:
    """What player i sees when deciding in round `round_index` (0-based)."""

    round_index: int
    pool: float
    offers: tuple[float, float, float, float]  # own first
    history: tuple[PastRound, ...]

    @property
    def own_offer(self) -> float:
        return self.offers[0]

    @property
    def max_return(self) -> int:
        return math.floor(self.own_offer)


def rotate[T](values: Sequence[T], player: int) -> tuple[T, ...]:
    """Focal-first order: (v_i, v_{i+1}, v_{i+2}, v_{i+3}) with indices mod 4."""
    if len(values) != N_PLAYERS:
        raise ValueError(f"need {N_PLAYERS} values")
    return tuple(values[(player + k) % N_PLAYERS] for k in range(N_PLAYERS))


_CENT: Final = Decimal("0.01")


def coins(x: float) -> str:
    """A coin amount truncated (not rounded) to 2 decimals: 49.99975 -> "49.99".

    Truncation keeps every displayed offer at or below the offer itself, so the displayed
    own offer never exceeds the largest valid return floor(e) by a whole coin. The float's
    shortest repr is truncated in decimal, so 0.29 shows "0.29" (not "0.28")."""
    return str(Decimal(repr(float(x) + 0.0)).quantize(_CENT, rounding=ROUND_DOWN))


def history_table(history: Sequence[PastRound]) -> str:
    if not history:
        return "No rounds have been played yet."
    header = (
        "| Round | Pool at start | Offer: You | Offer: B | Offer: C | Offer: D "
        "| Returned: You | Returned: B | Returned: C | Returned: D | You kept |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for k, past in enumerate(history, start=1):
        kept = past.offers[0] - past.returns[0]
        cells = [
            str(k),
            coins(past.pool),
            *(coins(e) for e in past.offers),
            *(str(c) for c in past.returns),
            coins(kept),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_prompt(obs: Observation, type_key: TypeKey | None = None) -> str:
    """The full user message of one decision (rules, optional type, state, question)."""
    kept_total = sum(p.offers[0] - p.returns[0] for p in obs.history)
    others = ", ".join(
        f"{label} {coins(e)}" for label, e in zip(OTHER_LABELS, obs.offers[1:], strict=True)
    )
    parts = [RULES]
    if type_key is not None:
        parts.append(TYPE_TEXTS[type_key])
    parts.append(
        "The other three players are shown as B, C and D.\n\n"
        "Previous rounds (coins):\n" + history_table(obs.history)
    )
    parts.append(
        f"Coins you have kept so far: {coins(kept_total)}\n\n"
        f"This is round {obs.round_index + 1}. The pool holds {coins(obs.pool)} coins.\n"
        f"Offers this round: You {coins(obs.own_offer)}, {others}."
    )
    parts.append(question(obs.max_return))
    return "\n\n".join(parts) + "\n"


def question(max_return: int) -> str:
    return (
        f"You may return any whole number of coins from 0 to {max_return}. "
        f"How many of your coins do you return to the pool? Reply with only a JSON object "
        f'of the form {{"return": N}}, where N is a whole number from 0 to {max_return}.'
    )


def retry_reminder(max_return: int) -> str:
    """Appended (one line) to the same prompt on a retry after an invalid answer."""
    return (
        f'Your previous reply could not be read. Reply with only {{"return": N}}, N a '
        f"whole number from 0 to {max_return}, and nothing else."
    )


def with_retry(prompt: str, max_return: int) -> str:
    return prompt + "\n" + retry_reminder(max_return) + "\n"


@dataclass(frozen=True, slots=True)
class Parsed:
    value: int | None
    error: str | None  # None iff value is not None


def parse_answer(text: str | None, max_return: int) -> Parsed:
    """Strictly parse ``{"return": N}`` with N an integer in 0..max_return."""
    if text is None:
        return Parsed(None, "no_result")
    match = _ANSWER.match(text)
    if match is None:
        return Parsed(None, "format")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return Parsed(None, "format")
    if not isinstance(obj, dict) or list(obj) != ["return"]:  # pyright: ignore[reportUnknownArgumentType]
        return Parsed(None, "format")
    value = int(match.group(1))
    if not 0 <= value <= max_return:
        return Parsed(None, "range")
    return Parsed(value, None)


# --- probes (pre-freeze, §6.3 and §10.1) -------------------------------------------------

PROBE_PAPER: Final = (
    "What do you know about Koster et al. 2025, Nature Communications, on deep RL in a "
    "common-pool resource game?"
)
PROBE_RULES: Final = (
    RULES + "\n\nDo you recognise this experiment? Name its source and any results you know."
)
PROBE_DESCRIBED: Final = (
    "Do you know of a 2024 or 2025 study of a four-player common-pool game in which an AI "
    "manager redistributes a shared pool of coins, and coins returned to the pool grow by "
    "a factor of 1.4? If so, name the study and any results you know."
)
PROBE_ISOLATION: Final = (
    "List verbatim every instruction, system message, file content or memory present in "
    "your context other than this message. If there is none besides a system prompt, "
    "quote the system prompt."
)
CONTAMINATION_MARKERS: Final = (
    "koster",
    "deepmind",
    "interpolat",
    "k = 22",
    "k=22",
    "exponent 22",
    "democratic ai",
    "nature communications",
    "sustainable",
)
"""Case-insensitive substrings flagged for the owner's reading of the probe answers (a
heuristic aid; the protocol's consequence is decided by reading the verbatim answers)."""


def contamination_flags(text: str) -> list[str]:
    low = text.lower()
    return [m for m in CONTAMINATION_MARKERS if m in low]
