"""Group-level fit/validation split of the human training games (protocol 6.1; pure).

Units are the 200 independent training groups: one game for Exps 1-2, and the three
consecutive games of an Exp 4 group (which share their `launch_id`) as one unit. The
units are stratified by (mechanism, cohort) in the order of `config.SPLIT_STRATA`
(40 units each). One generator `numpy.random.default_rng(20261020)` draws, stratum by
stratum in that order, a permutation of the stratum's units sorted by `launch_id`
(Exp 4: the Game 1 `launch_id`, which every game of the group shares); the first 32
permuted units form the fit split and the last 8 the validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from phasespace.citizens import config
from phasespace.citizens.records import TRAINING_MECHANISMS, Games

type BoolArray = npt.NDArray[np.bool_]

FIT: Final = "fit"
VALIDATION: Final = "validation"


@dataclass(frozen=True, slots=True)
class Unit:
    """One independent group: its stratum, its key (`launch_id`) and its split."""

    mechanism: str
    cohort: str
    launch_id: str
    split: str  # "fit" or "validation"


@dataclass(frozen=True, slots=True)
class Split:
    units: tuple[Unit, ...]

    def unit_of(self, games: Games) -> npt.NDArray[np.int64]:
        """(G,) index into `units` of every game (raises if a game has no unit)."""
        index = {(u.mechanism, u.cohort, u.launch_id): k for k, u in enumerate(self.units)}
        keys = zip(
            games.mechanism.tolist(), games.cohort.tolist(), games.launch_id.tolist(), strict=True
        )
        try:
            return np.array([index[key] for key in keys], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"game {exc.args[0]} belongs to no split unit") from exc

    def mask(self, games: Games, split: str) -> BoolArray:
        """(G,) True for the games whose unit is in `split`."""
        if split not in (FIT, VALIDATION):
            raise ValueError(f"unknown split {split!r}")
        unit_split = np.array([u.split == split for u in self.units], dtype=np.bool_)
        return unit_split[self.unit_of(games)]

    def to_rows(self) -> list[list[str]]:
        return [[u.mechanism, u.cohort, u.launch_id, u.split] for u in self.units]


def human_training_games(games: Games) -> Games:
    """The human Equal/Mixed/Proportional games (refuses anything else)."""
    human = games.select(games.is_human)
    if not np.isin(human.mechanism, TRAINING_MECHANISMS).all():
        raise ValueError("human games outside the training mechanisms were passed")
    return human


def make_split(
    games: Games,
    *,
    seed: int = config.SPLIT_SEED,
    n_fit: int = config.FIT_UNITS_PER_STRATUM,
    n_val: int = config.VAL_UNITS_PER_STRATUM,
) -> Split:
    """The protocol 6.1 split of the human training games in `games`."""
    human = human_training_games(games)
    rng = np.random.default_rng(seed)
    units: list[Unit] = []
    covered = 0
    for mechanism, cohort in config.SPLIT_STRATA:
        stratum = human.where(mechanism=mechanism, cohort=cohort)
        covered += stratum.n_games
        keys = sorted(set(stratum.launch_id.tolist()))
        if len(keys) != n_fit + n_val:
            raise ValueError(
                f"stratum {mechanism} / {cohort} has {len(keys)} units, expected {n_fit + n_val}"
            )
        if cohort == "Exp 4":
            for key in keys:
                games_k = stratum.select(stratum.launch_id == key).exp4_game
                if sorted(games_k.tolist()) != [1, 2, 3]:
                    raise ValueError(f"Exp 4 group {key} does not hold Games 1-3")
        perm = rng.permutation(len(keys))
        for rank, k in enumerate(perm.tolist()):
            split = FIT if rank < n_fit else VALIDATION
            units.append(Unit(mechanism, cohort, keys[k], split))
    if covered != human.n_games:
        raise ValueError("some human training games fall outside the split strata")
    return Split(units=tuple(units))
