"""Game-level metrics exactly as defined in the published notebook (pure).

Source: google-deepmind/sustainable_behavior, notebooks/sustainable_behavior.ipynb
(functions `calculate_gini`, `get_surplus_and_ginis`, `find_first_depletion_trial`,
`get_depletion_and_active_players`; statistics with `scipy.stats.ranksums`). The
notebook applies no row or game filters: it uses every row of a
`mech_name_by_player` subset, grouped by `launch_id`, including rounds after
depletion. The published CSV already contains only the analysed games (every human
row has `four_usable_players`, `completed` and `first_x_games` True).

- Surplus of a game: the mean over the 4 players of each player's mean reward per
  round (`player_reward_i`, i.e. e - c), so it is per player per round.
- Gini of a game: 0.5 * mean_{i,j} |x_i - x_j| / mean(x) over the 4 players' mean
  rewards, the mean running over all 16 ordered pairs including i = j (so the maximum
  for 4 players is 0.75). The paper calls it the Gini; it is not the CSV's per-round
  `offer_gini` column.
- Depletion round: 1-based index of the first round whose pool before allocation is
  below 1, else 40 (the notebook returns 40 even for Exp 4 games of other lengths).
- Players active: the mean over rounds of `players_kept_active`, and its value in
  the game's last round.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import stats

from phasespace.citizens.game import DEPLETION_THRESHOLD, N_ROUNDS

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

NOT_DEPLETED: Final = N_ROUNDS
"""The notebook's depletion round of a game whose pool never fell below 1."""


def gini(values: npt.ArrayLike) -> FloatArray:
    """Notebook Gini over the last axis: 0.5 * mean|x_i - x_j| / mean(x), x >= 0.

    Convention: an all-zero vector (mean 0, so 0/0 in the notebook formula) has Gini
    0.0, perfect equality, since every player received the same (nothing). Rewards are
    non-negative, so mean(x) = 0 only for all-zero vectors; negative values are rejected.
    """
    x = np.asarray(values, dtype=np.float64)
    if np.any(x < 0.0):
        raise ValueError("the Gini is defined here for non-negative values only")
    mad = np.abs(x[..., :, None] - x[..., None, :]).mean(axis=(-2, -1))
    mean = x.mean(axis=-1)
    return np.divide(0.5 * mad, mean, out=np.zeros_like(mad), where=mean > 0.0)


def _round_mask(n_rounds: npt.ArrayLike, length: int) -> npt.NDArray[np.bool_]:
    n = np.asarray(n_rounds, dtype=np.int64)
    return np.arange(length)[None, :] < n[:, None]


def mean_rewards(rewards: npt.ArrayLike, n_rounds: npt.ArrayLike) -> FloatArray:
    """(G, 4) each player's mean reward per round over the game's recorded rounds."""
    r = np.asarray(rewards, dtype=np.float64)
    mask = _round_mask(n_rounds, r.shape[1])[..., None]
    return np.where(mask, r, 0.0).sum(axis=1) / mask.sum(axis=1)


def surplus(rewards: npt.ArrayLike, n_rounds: npt.ArrayLike) -> FloatArray:
    """(G,) mean over players of each player's mean reward per round."""
    return mean_rewards(rewards, n_rounds).mean(axis=-1)


def reward_gini(rewards: npt.ArrayLike, n_rounds: npt.ArrayLike) -> FloatArray:
    """(G,) notebook Gini of the players' mean rewards."""
    return gini(mean_rewards(rewards, n_rounds))


def depletion_round(pool: npt.ArrayLike, n_rounds: npt.ArrayLike) -> IntArray:
    """(G,) 1-based first round with pool < 1 (before allocation), else 40."""
    p = np.asarray(pool, dtype=np.float64)
    below = _round_mask(n_rounds, p.shape[1]) & (p < DEPLETION_THRESHOLD)
    return np.where(below.any(axis=1), below.argmax(axis=1) + 1, NOT_DEPLETED).astype(np.int64)


def mean_players_active(players_kept_active: npt.ArrayLike, n_rounds: npt.ArrayLike) -> FloatArray:
    """(G,) mean over recorded rounds of `players_kept_active`."""
    a = np.asarray(players_kept_active, dtype=np.float64)
    mask = _round_mask(n_rounds, a.shape[1])
    return np.where(mask, a, 0.0).sum(axis=1) / mask.sum(axis=1)


def last_players_active(players_kept_active: npt.ArrayLike, n_rounds: npt.ArrayLike) -> IntArray:
    """(G,) `players_kept_active` in each game's last recorded round."""
    a = np.asarray(players_kept_active, dtype=np.int64)
    n = np.asarray(n_rounds, dtype=np.int64)
    return a[np.arange(a.shape[0]), n - 1]


def rank_sum_z(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Wilcoxon rank-sum statistic z of `scipy.stats.ranksums(a, b)` (games as units)."""
    return float(
        stats.ranksums(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)).statistic
    )
