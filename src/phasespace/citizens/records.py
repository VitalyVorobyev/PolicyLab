"""Per-game arrays of recorded games (the parsed dataset), independent of any I/O."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]
type FlagArray = npt.NDArray[np.int8]
type StrArray = npt.NDArray[np.str_]

HUMAN_COHORTS: Final = ("Exp 1", "Exp 2", "Exp 3", "Exp 4")
BC_COHORTS: Final = ("BC 1", "BC 2")
TRAINING_MECHANISMS: Final = ("Equal Baseline", "Mixed Baseline", "Proportional Baseline")
HELDOUT_MECHANISMS: Final = ("Interpolating Baseline", "RL Agent (M1)", "RL Agent (M2)")
MECHANISMS: Final = TRAINING_MECHANISMS + HELDOUT_MECHANISMS


@dataclass(frozen=True, slots=True)
class Games:
    """Per-game arrays, games x rounds (x players); rounds past `n_rounds` are padding.

    Float padding is NaN, integer padding -1. Tri-state flags (CSV "True"/"False"/empty)
    are int8 1/0/-1; BC rows leave every human-only column empty (-1 or NaN).
    """

    # per game
    mechanism: StrArray  # (G,) base mechanism family (no Exp 4 " Game k" suffix)
    cohort: StrArray  # (G,) "Exp 1".."Exp 4", "BC 1", "BC 2"
    exp4_game: IntArray  # (G,) 1..3 for Exp 4, else 0
    launch_id: StrArray  # (G,) as written in the CSV
    n_rounds: IntArray  # (G,)
    first_row: IntArray  # (G,) 0-based data-row index of round 0 in the CSV
    # per game and round
    pool: FloatArray  # (G, T) R_t before allocation (mechanism_observation.pool)
    next_pool: FloatArray  # (G, T) next_environment_state.pool (human rows only)
    mechanism_reward: FloatArray  # (G, T)
    offer_gini: FloatArray  # (G, T) the CSV's per-round Gini of offers
    uniform_offer_ratio: FloatArray  # (G, T) mechanism_parameters_uniform_offer_ratio (w)
    distribute_pool_ratio: FloatArray  # (G, T) mechanism_parameters_distribute_pool_ratio
    players_kept_active: IntArray  # (G, T)
    # per game, round and player
    offers: FloatArray  # (G, T, 4) e_{i,t} (offer_i)
    actions: FloatArray  # (G, T, 4) c_{i,t} (player_action_i)
    rewards: FloatArray  # (G, T, 4) player_reward_i (= e - c)
    prev_reciprocations: FloatArray  # (G, T, 4) mechanism_observation.prev_reciprocations
    prev_offers: FloatArray  # (G, T, 4) mechanism_observation.prev_offers
    next_reciprocations: FloatArray  # (G, T, 4) next_environment_state.reciprocations
    # flags (G, T) and (G, T, 4)
    four_usable_players: FlagArray
    completed: FlagArray
    first_x_games: FlagArray
    four_players_present: FlagArray
    game_has_no_pool: FlagArray
    player_present: FlagArray
    player_has_no_choice: FlagArray

    @property
    def n_games(self) -> int:
        return int(self.n_rounds.shape[0])

    @property
    def is_human(self) -> BoolArray:
        return np.isin(self.cohort, HUMAN_COHORTS)

    def round_mask(self) -> BoolArray:
        """(G, T) True for recorded rounds."""
        t = np.arange(int(self.pool.shape[1]), dtype=np.int64)
        return t[None, :] < self.n_rounds[:, None]

    def select(self, mask: BoolArray) -> Games:
        """The games where `mask` (G,) is True."""
        return Games(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    def where(
        self,
        *,
        mechanism: str | None = None,
        cohort: str | None = None,
        exp4_game: int | None = None,
    ) -> Games:
        mask = np.ones(self.n_games, dtype=np.bool_)
        if mechanism is not None:
            mask &= self.mechanism == mechanism
        if cohort is not None:
            mask &= self.cohort == cohort
        if exp4_game is not None:
            mask &= self.exp4_game == exp4_game
        return self.select(mask)


def field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(Games))
