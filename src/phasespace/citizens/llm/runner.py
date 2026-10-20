"""Drive B4 games with LLM players, archive every call, and write runs.

Each player of each round with an offer of at least 1 coin is one fresh call (a player
offered less than 1 coin is excluded and returns 0 without a call). The four calls of
a round run in parallel; several games run at once; a semaphore caps the concurrent
calls (default 8). An invalid answer (unparseable or out of range) is retried up to
twice with a one-line format reminder; a decision still invalid after the retries makes
the game unrecoverable: it is discarded (counted, archived) and replaced by the next
game of the pre-declared order, up to the discard limit (10 % of the target, at least 1).

**Infrastructure errors are not invalid answers** (protocol-validation A27). A timeout,
non-zero exit, CLI/API error (rate limits included), missing result or a reported model
other than the requested one (A26) is retried with the same prompt after waits of
5, 10, 20, 40, 80 and 160 s (`config.B4_INFRA_BACKOFF_S`); it never counts toward the
discard rule. If a decision exhausts these retries the run stops with
`InfrastructureError`; re-running the same command resumes it, replaying the
interrupted games from their first round (their calls are archived under abandoned/).

**Resume safety.** Work happens in ``<runs-dir>/_work/<run-id>/``. Calls of a game in
progress are appended to ``game-NNNN.calls.partial.jsonl``; when the game ends that file
is renamed to ``game-NNNN.calls.jsonl`` and then ``game-NNNN.json`` is written
atomically (temporary file + rename): the JSON file marks a finished game. A rerun with
the same configuration (``config.json`` must match exactly) skips finished games and
moves the calls of an interrupted game to ``abandoned/`` (kept, and counted in the call
total). ``finalize`` then writes the immutable run ``<runs-dir>/<run-id>/``.

**Privacy.** Raw call records (prompts, raw CLI output) can contain the operator's
account e-mail address, which the CLI injects into the model's environment context. They
stay local under ``runs/`` (git-ignored) and are never published; published B4 data are
the parsed actions only (``games.json``, ``rounds.json``).

Commands::

    uv run python -m phasespace.citizens.llm.runner probe --run-id <id>
    uv run python -m phasespace.citizens.llm.runner pilot --run-id <id> [--games 2]
    uv run python -m phasespace.citizens.llm.runner calibrate --run-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt

from phasespace.citizens import config as cfg
from phasespace.citizens import data, game, metrics
from phasespace.citizens import export as c1
from phasespace.citizens.llm import prompts
from phasespace.citizens.llm.backend import (
    Backend,
    CallResult,
    ClaudeCLI,
    infrastructure_error,
    passed_environment_names,
)
from phasespace.citizens.llm.prompts import Observation, PastRound, TypeKey
from phasespace.citizens.mechanisms import W_PROPORTIONAL, baseline_offers
from phasespace.gravity.export import (
    dumps_json,
    environment_info,
    git_source,
    validate_run_id,
    write_run_directory,
)

type FloatArray = npt.NDArray[np.float64]
type Variant = Literal["plain", "calibrated"]

WORKERS: Final = 8
MAX_RETRIES: Final = cfg.B4_RETRIES
DISCARD_LIMIT: Final = cfg.B4_DISCARD_LIMIT
B4_SEED: Final = cfg.B4_SEED
CALL_CEILING: Final = cfg.B4_CALL_CEILING
WALL_CEILING_DAYS: Final = 14.0
PILOT_GAMES: Final = 2
PROBE_REPEATS: Final = 5
CALIBRATION_GAMES_PER_COMPOSITION: Final = cfg.B4_CALIBRATION_GAMES_PER_COMPOSITION
EVALUATION_GAMES: Final = cfg.B4_EVAL_GAMES
EVALUATION_VARIANTS: Final = 2
EVALUATION_MECHANISMS: Final = 3
WORK_DIR: Final = "_work"
INFRA_BACKOFF_S: Final = cfg.B4_INFRA_BACKOFF_S
"""Waits before infrastructure retries 1..6 of one decision (A27)."""
INFRASTRUCTURE: Final = "infrastructure"
INVALID_ANSWER: Final = "invalid_answer"
LEGACY_INFRASTRUCTURE_KINDS: Final = frozenset(
    {"timeout", "cli_error", "nonzero_exit", "api_error", "no_result", "model_mismatch",
     "unreadable_output"}
)  # fmt: skip
"""Error kinds that are infrastructure errors (records written before A27 carry no
`error_class`; their `timeout` / `cli_error` / `no_result` kinds are classified here)."""


class InfrastructureError(RuntimeError):
    """A decision exhausted its infrastructure retries: the run stops (resume later)."""


# --- mechanisms for single LLM games -----------------------------------------------------


class GameMechanism(Protocol):
    """Offers for one game; `state` is whatever the mechanism carries between rounds."""

    @property
    def name(self) -> str: ...

    def start(self) -> object: ...

    def offers(
        self,
        state: object,
        pool: float,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
    ) -> tuple[FloatArray, object]: ...


@dataclass(frozen=True, slots=True)
class BaselineMechanism:
    """A baseline rule with fixed weight w under the deployed offer rule (D33)."""

    name: str
    w: float

    def start(self) -> object:
        return None

    def offers(
        self,
        state: object,
        pool: float,
        prev_offers: FloatArray | None,
        prev_reciprocations: FloatArray | None,
    ) -> tuple[FloatArray, object]:
        del prev_offers
        return np.asarray(baseline_offers(pool, prev_reciprocations, self.w)), state


PROPORTIONAL: Final = BaselineMechanism("Proportional Baseline", W_PROPORTIONAL)


# --- one decision ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    value: int | None  # None: unrecoverable
    attempts: list[dict[str, Any]]


def error_class(record: dict[str, Any]) -> str | None:
    """None for a valid answer, else INFRASTRUCTURE or INVALID_ANSWER (A27)."""
    if record.get("value") is not None:
        return None
    stored = record.get("error_class")
    if stored is not None:
        return str(stored)
    return INFRASTRUCTURE if record.get("error") in LEGACY_INFRASTRUCTURE_KINDS else INVALID_ANSWER


async def decide(
    prompt: str,
    max_return: int,
    backend: Backend,
    semaphore: asyncio.Semaphore,
    meta: dict[str, Any],
    sink: Callable[[dict[str, Any]], None],
    *,
    max_retries: int = MAX_RETRIES,
    infra_backoff: Sequence[float] = INFRA_BACKOFF_S,
) -> Decision:
    """Ask, parse strictly, retry; every attempt goes to `sink`.

    An invalid answer is retried with the format reminder up to `max_retries` times,
    then the decision is unrecoverable (value None). An infrastructure error (A27) is
    retried with the same text after `infra_backoff[k]` seconds, k = 0, 1, ...; when the
    waits are used up, `InfrastructureError` stops the run.
    """
    attempts: list[dict[str, Any]] = []
    invalid = infra = 0
    while True:
        text = prompt if invalid == 0 else prompts.with_retry(prompt, max_return)
        started = datetime.now(UTC).isoformat()
        async with semaphore:
            result: CallResult = await backend.complete(text)
        kind = result.infra_error or infrastructure_error(result, None)
        if kind is not None:
            parsed, cls = prompts.Parsed(None, kind), INFRASTRUCTURE
        else:
            parsed = prompts.parse_answer(result.result_text, max_return)
            cls = None if parsed.value is not None else INVALID_ANSWER
        record = {
            **meta,
            "attempt": len(attempts),
            "started_at": started,
            "prompt": text,
            "max_return": max_return,
            "value": parsed.value,
            "error": parsed.error,
            "error_class": cls,
            **result.to_json(),
        }
        attempts.append(record)
        sink(record)
        if parsed.value is not None:
            return Decision(parsed.value, attempts)
        if cls == INFRASTRUCTURE:
            if infra >= len(infra_backoff):
                raise InfrastructureError(
                    f"{meta}: {infra + 1} infrastructure errors in a row (last: {kind}); "
                    "the run stops and resumes with the same command"
                )
            await asyncio.sleep(infra_backoff[infra])
            infra += 1
            continue
        invalid += 1
        if invalid > max_retries:
            return Decision(None, attempts)


# --- one game --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GameSpec:
    index: int  # position in the pre-declared order
    n_rounds: int
    types: tuple[TypeKey | None, TypeKey | None, TypeKey | None, TypeKey | None]

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "n_rounds": self.n_rounds, "types": list(self.types)}


@dataclass(frozen=True, slots=True)
class RoundRecord:
    pool: float
    offers: list[float]
    returns: list[int]
    next_pool: float
    called: list[bool]  # whether each player was asked (offer >= 1)


@dataclass(frozen=True, slots=True)
class GameOutcome:
    index: int
    status: Literal["complete", "discarded"]
    reason: str | None
    spec: GameSpec
    rounds: list[RoundRecord]
    attempts: int
    invalid_attempts: int  # invalid answers only (A27)
    seconds: float
    infrastructure_attempts: int = 0  # A27: never a discard

    def metrics(self) -> dict[str, float] | None:
        if self.status != "complete":
            return None
        return game_metrics(self.rounds)

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "reason": self.reason,
            "spec": self.spec.to_json(),
            "attempts": self.attempts,
            "invalid_attempts": self.invalid_attempts,
            "infrastructure_attempts": self.infrastructure_attempts,
            "seconds": self.seconds,
            "metrics": self.metrics(),
            "rounds": [
                {
                    "pool": r.pool,
                    "offers": r.offers,
                    "returns": r.returns,
                    "next_pool": r.next_pool,
                    "called": r.called,
                }
                for r in self.rounds
            ],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> GameOutcome:
        s = d["spec"]
        return GameOutcome(
            index=int(d["index"]),
            status=d["status"],
            reason=d["reason"],
            spec=GameSpec(int(s["index"]), int(s["n_rounds"]), tuple(s["types"])),  # pyright: ignore[reportArgumentType]
            rounds=[
                RoundRecord(
                    float(r["pool"]),
                    [float(v) for v in r["offers"]],
                    [int(v) for v in r["returns"]],
                    float(r["next_pool"]),
                    [bool(v) for v in r["called"]],
                )
                for r in d["rounds"]
            ],
            attempts=int(d["attempts"]),
            invalid_attempts=int(d["invalid_attempts"]),
            seconds=float(d["seconds"]),
            infrastructure_attempts=int(d.get("infrastructure_attempts", 0)),
        )


def game_metrics(rounds: Sequence[RoundRecord]) -> dict[str, float]:
    """The four §8 metrics of one finished game (notebook definitions)."""
    offers = np.array([[r.offers for r in rounds]])
    returns = np.array([[r.returns for r in rounds]], dtype=np.float64)
    pool = np.array([[r.pool for r in rounds]])
    n = np.array([len(rounds)])
    rewards = game.surplus(offers, returns)
    last = np.asarray(offers[0, -1])
    return {
        "surplus": float(metrics.surplus(rewards, n)[0]),
        "gini": float(metrics.reward_gini(rewards, n)[0]),
        "depletion_round": float(metrics.depletion_round(pool, n)[0]),
        "players_active_end": float(np.count_nonzero(last >= game.EXCLUSION_THRESHOLD)),
    }


def observation(
    player: int, round_index: int, pool: float, offers: FloatArray, history: Sequence[RoundRecord]
) -> Observation:
    """What `player` sees: current offers and every past round, rotated focal-first."""

    def rot4(v: Sequence[float]) -> tuple[float, float, float, float]:
        a, b, c, d = prompts.rotate([float(x) for x in v], player)
        return (a, b, c, d)

    def rot4i(v: Sequence[int]) -> tuple[int, int, int, int]:
        a, b, c, d = prompts.rotate(list(v), player)
        return (a, b, c, d)

    past = tuple(PastRound(r.pool, rot4(r.offers), rot4i(r.returns)) for r in history)
    return Observation(round_index, float(pool), rot4(offers.tolist()), past)


async def play_game(
    spec: GameSpec,
    mechanism: GameMechanism,
    backend: Backend,
    semaphore: asyncio.Semaphore,
    sink: Callable[[dict[str, Any]], None],
    *,
    max_retries: int = MAX_RETRIES,
    infra_backoff: Sequence[float] = INFRA_BACKOFF_S,
) -> GameOutcome:
    """Play one game to its last round, or until a decision is unrecoverable."""
    start = time.perf_counter()
    state = game.GameState()
    mech_state = mechanism.start()
    history: list[RoundRecord] = []
    prev_offers: FloatArray | None = None
    attempts = invalid = infra = 0
    for t in range(spec.n_rounds):
        offers, mech_state = mechanism.offers(
            mech_state,
            state.pool,
            prev_offers,
            None if state.prev_reciprocations is None else np.array(state.prev_reciprocations),
        )
        called = [bool(e >= game.EXCLUSION_THRESHOLD) for e in offers]
        jobs: list[asyncio.Task[Decision]] = []
        for i in range(game.N_PLAYERS):
            if not called[i]:
                continue
            obs = observation(i, t, state.pool, offers, history)
            meta = {"game": spec.index, "round": t, "player": i, "type": spec.types[i]}
            text = prompts.render_prompt(obs, spec.types[i])
            jobs.append(
                asyncio.create_task(
                    decide(
                        text,
                        obs.max_return,
                        backend,
                        semaphore,
                        meta,
                        sink,
                        max_retries=max_retries,
                        infra_backoff=infra_backoff,
                    )
                )
            )
        decisions = await asyncio.gather(*jobs)
        for d in decisions:
            attempts += len(d.attempts)
            invalid += sum(1 for a in d.attempts if error_class(a) == INVALID_ANSWER)
            infra += sum(1 for a in d.attempts if error_class(a) == INFRASTRUCTURE)
        if any(d.value is None for d in decisions):
            return GameOutcome(
                spec.index,
                "discarded",
                f"unrecoverable decision in round {t}",
                spec,
                history,
                attempts,
                invalid,
                time.perf_counter() - start,
                infra,
            )
        values = iter(d.value for d in decisions)
        returns = [int(next(values) or 0) if called[i] else 0 for i in range(game.N_PLAYERS)]
        step = game.step(state, offers, returns, integer=True)
        history.append(
            RoundRecord(
                pool=state.pool,
                offers=[float(v) for v in offers],
                returns=returns,
                next_pool=step.state.pool,
                called=called,
            )
        )
        prev_offers = offers
        state = step.state
    return GameOutcome(
        spec.index,
        "complete",
        None,
        spec,
        history,
        attempts,
        invalid,
        time.perf_counter() - start,
        infra,
    )


# --- resumable runs ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything that defines a B4 run; must be identical when resuming."""

    run_id: str
    label: str  # e.g. "pilot"
    variant: Variant
    mechanism: str
    target_games: int
    n_rounds: int
    workers: int
    max_retries: int
    discard_limit: int
    seed: int
    backend: dict[str, Any]
    prompt_version: str = prompts.PROMPT_VERSION
    system_prompt: str = prompts.SYSTEM_PROMPT
    rules_sha256: str = field(
        default_factory=lambda: hashlib.sha256(prompts.RULES.encode("utf-8")).hexdigest()
    )
    type_texts: dict[str, str] = field(
        default_factory=lambda: {str(k): v for k, v in prompts.TYPE_TEXTS.items()}
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "variant": self.variant,
            "mechanism": self.mechanism,
            "target_games": self.target_games,
            "n_rounds": self.n_rounds,
            "workers": self.workers,
            "max_retries": self.max_retries,
            "discard_limit": self.discard_limit,
            "seed": self.seed,
            "backend": self.backend,
            "prompt_version": self.prompt_version,
            "system_prompt": self.system_prompt,
            "rules_sha256": self.rules_sha256,
            "type_texts": self.type_texts,
        }


def discard_limit(target_games: int) -> int:
    """Discards allowed before the run stops: 10 % of the target, at least 1."""
    return max(1, math.floor(DISCARD_LIMIT * target_games))


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write via a temporary sibling + rename; refuses to replace an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; finished game files are never overwritten")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("xb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(tmp, path)


class GameStore:
    """Per-game files in the work directory (see the module docstring)."""

    def __init__(self, work: Path) -> None:
        self.work = work
        self.abandoned = work / "abandoned"

    def outcome_path(self, index: int) -> Path:
        return self.work / f"game-{index:04d}.json"

    def calls_path(self, index: int) -> Path:
        return self.work / f"game-{index:04d}.calls.jsonl"

    def partial_path(self, index: int) -> Path:
        return self.work / f"game-{index:04d}.calls.partial.jsonl"

    def finished(self, index: int) -> GameOutcome | None:
        path = self.outcome_path(index)
        if not path.exists():
            return None
        return GameOutcome.from_json(json.loads(path.read_text("utf-8")))

    def clear_interrupted(self, index: int) -> None:
        """Move calls of an interrupted attempt at game `index` to abandoned/."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        for path in (self.partial_path(index), self.calls_path(index)):
            if path.exists():
                self.abandoned.mkdir(exist_ok=True)
                os.rename(path, self.abandoned / f"{path.name}.{stamp}")

    def sink(self, index: int) -> tuple[Callable[[dict[str, Any]], None], Callable[[], None]]:
        fh = self.partial_path(index).open("a", encoding="utf-8")

        def write(record: dict[str, Any]) -> None:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()

        return write, fh.close

    def finish(self, outcome: GameOutcome) -> None:
        partial, calls = self.partial_path(outcome.index), self.calls_path(outcome.index)
        if calls.exists():
            raise FileExistsError(f"{calls} exists")
        os.rename(partial, calls)
        _write_atomic(
            self.outcome_path(outcome.index),
            (json.dumps(outcome.to_json(), sort_keys=True, indent=1) + "\n").encode("utf-8"),
        )


def prepare_work_dir(work: Path, config: RunConfig) -> GameStore:
    """Create the work directory with config.json, or check it matches on resume."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / "config.json"
    current = config.to_json()
    if path.exists():
        stored: dict[str, Any] = json.loads(path.read_text("utf-8"))
        if stored != json.loads(json.dumps(current)):
            raise ValueError(
                f"{path} differs from this invocation's configuration; a run is resumed "
                "only with identical settings"
            )
    else:
        _write_atomic(path, (json.dumps(current, sort_keys=True, indent=1) + "\n").encode())
    return GameStore(work)


async def run_games(
    config: RunConfig,
    specs: Callable[[int], GameSpec],
    mechanism: GameMechanism,
    backend: Backend,
    store: GameStore,
    *,
    log: Callable[[str], None] | None = None,
    infra_backoff: Sequence[float] = INFRA_BACKOFF_S,
) -> list[GameOutcome]:
    """Play games 0, 1, ... until `target_games` are complete or discards exceed the limit.

    `specs(k)` gives the k-th game of the pre-declared order; finished games are loaded,
    not replayed. Game concurrency is workers // 2 (at least 1); call concurrency is
    `workers`.
    """
    semaphore = asyncio.Semaphore(config.workers)
    game_slots = asyncio.Semaphore(max(1, config.workers // 2))
    outcomes: dict[int, GameOutcome] = {}

    async def one(index: int) -> None:
        done = store.finished(index)
        if done is not None:
            outcomes[index] = done
            return
        async with game_slots:
            store.clear_interrupted(index)
            write, close = store.sink(index)
            try:
                outcome = await play_game(
                    specs(index),
                    mechanism,
                    backend,
                    semaphore,
                    write,
                    max_retries=config.max_retries,
                    infra_backoff=infra_backoff,
                )
            finally:
                close()
            store.finish(outcome)
            outcomes[index] = outcome
            if log is not None:
                log(
                    f"game {index}: {outcome.status} ({outcome.attempts} calls, "
                    f"{outcome.seconds:.0f} s) {outcome.metrics()}"
                )

    next_index = 0
    while True:
        complete = sum(1 for o in outcomes.values() if o.status == "complete")
        discarded = sum(1 for o in outcomes.values() if o.status == "discarded")
        if complete >= config.target_games or discarded > config.discard_limit:
            break
        batch = list(range(next_index, next_index + config.target_games - complete))
        next_index += len(batch)
        await asyncio.gather(*(one(k) for k in batch))
    return [outcomes[k] for k in sorted(outcomes)]


# --- type assignment (B4-calibrated) ----------------------------------------------------


def compositions() -> list[tuple[int, int, int]]:
    """The 15 type compositions (n_T1, n_T2, n_T3) of 4 players."""
    return [(a, b, 4 - a - b) for a in range(5) for b in range(5 - a)]


def calibration_specs(
    n_rounds: int = game.N_ROUNDS,
    games_per_composition: int = CALIBRATION_GAMES_PER_COMPOSITION,
    seed: int = B4_SEED,
) -> list[GameSpec]:
    """60 calibration games (4 per composition) in a seeded order with seeded seats."""
    rng = np.random.default_rng(seed)
    games: list[list[TypeKey]] = []
    for a, b, c in compositions():
        counts = (a, b, c)
        base: list[TypeKey] = [
            t for t, n in zip(prompts.TYPE_KEYS, counts, strict=True) for _ in range(n)
        ]
        games.extend(list(base) for _ in range(games_per_composition))
    order = rng.permutation(len(games))
    out: list[GameSpec] = []
    for k, g in enumerate(order):
        seats: list[TypeKey] = [games[int(g)][int(j)] for j in rng.permutation(game.N_PLAYERS)]
        out.append(GameSpec(k, n_rounds, (seats[0], seats[1], seats[2], seats[3])))
    return out


def replacement_specs(base: Sequence[GameSpec], store: GameStore) -> Callable[[int], GameSpec]:
    """Game k of a fixed design `base`, with replacements beyond it.

    Game k < len(base) is base[k]. Game len(base) + j replaces the j-th discarded game
    (in index order) with the same seats and types under a new index, so a discard never
    changes the design. `run_games` asks for a replacement only after the batch holding
    the discarded game has finished, so the discarded games are known by then.
    """
    n = len(base)

    def spec(k: int) -> GameSpec:
        if k < n:
            return base[k]
        discarded: list[GameOutcome] = []
        for i in range(k):
            done = store.finished(i)
            if done is not None and done.status == "discarded":
                discarded.append(done)
        j = k - n
        if j >= len(discarded):
            raise ValueError(f"game {k} replaces no discarded game")
        replaced = discarded[j].spec
        return GameSpec(k, replaced.n_rounds, replaced.types)

    return spec


def evaluation_types(
    weights: Sequence[float], index: int, seed: int = B4_SEED
) -> tuple[TypeKey, TypeKey, TypeKey, TypeKey]:
    """Each player's type drawn i.i.d. from `weights` over (T1, T2, T3); game-keyed rng."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (3,) or np.any(w < 0) or not math.isclose(float(w.sum()), 1.0):
        raise ValueError("weights must be 3 non-negative numbers summing to 1")
    draw = np.random.default_rng([seed, index]).choice(3, size=game.N_PLAYERS, p=w)
    t: list[TypeKey] = [prompts.TYPE_KEYS[int(k)] for k in draw]
    return (t[0], t[1], t[2], t[3])


# --- statistics and budget ---------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def call_statistics(records: Sequence[dict[str, Any]], wall_s: float | None) -> dict[str, Any]:
    latencies = np.array([float(r["latency_s"]) for r in records]) if records else np.zeros(0)
    errors: dict[str, int] = {}
    for r in records:
        if r["error"] is not None:
            errors[str(r["error"])] = errors.get(str(r["error"]), 0) + 1
    classes = [error_class(r) for r in records]
    first = [r for r in records if r["attempt"] == 0]
    costs = [float(r["cost_usd"]) for r in records if r.get("cost_usd") is not None]
    models = sorted({m for r in records for m in r.get("models", [])})
    stats: dict[str, Any] = {
        "calls": len(records),
        "decisions": len(first),
        "invalid_attempts": classes.count(INVALID_ANSWER),
        "infrastructure_attempts": classes.count(INFRASTRUCTURE),
        "invalid_first_attempts": sum(1 for r in first if error_class(r) == INVALID_ANSWER),
        "retries": len(records) - len(first),
        "errors_by_kind": errors,
        "latency_mean_s": float(latencies.mean()) if latencies.size else None,
        "latency_median_s": float(np.median(latencies)) if latencies.size else None,
        "latency_p90_s": float(np.percentile(latencies, 90)) if latencies.size else None,
        "latency_max_s": float(latencies.max()) if latencies.size else None,
        "cost_usd_total": float(sum(costs)),
        "models_reported": models,
    }
    if wall_s is not None and wall_s > 0:
        stats["wall_s"] = wall_s
        stats["throughput_calls_per_hour"] = 3600.0 * len(records) / wall_s
    return stats


def budget_projection(
    calls_per_game: float,
    retry_fraction: float,
    throughput_per_hour: float | None,
    *,
    probe_calls: int,
    pilot_calls: int,
) -> dict[str, Any]:
    """Planned B4 calls (§7) at the measured calls per game, against the ceilings."""
    calib_games = len(compositions()) * CALIBRATION_GAMES_PER_COMPOSITION
    eval_games = EVALUATION_VARIANTS * EVALUATION_MECHANISMS * EVALUATION_GAMES
    per_game_max = game.N_PLAYERS * game.N_ROUNDS
    measured = (calib_games + eval_games) * calls_per_game
    worst = (calib_games + eval_games) * per_game_max * (1.0 + retry_fraction)
    total = probe_calls + pilot_calls + measured
    total_worst = probe_calls + pilot_calls + worst
    out: dict[str, Any] = {
        "calibration_games": calib_games,
        "evaluation_games": eval_games,
        "calls_per_game_measured": calls_per_game,
        "calls_per_game_max": per_game_max,
        "retry_fraction_measured": retry_fraction,
        "projected_calls_at_measured_rate": total,
        "projected_calls_worst_case": total_worst,
        "call_ceiling": CALL_CEILING,
        "within_ceiling_measured": total <= CALL_CEILING,
        "within_ceiling_worst_case": total_worst <= CALL_CEILING,
        "note": "calls per game measured under Proportional; held-out mechanisms that "
        "exclude players (offers < 1) need fewer calls, so the worst case is 160 per game "
        "plus retries at the measured rate",
    }
    if throughput_per_hour:
        out["throughput_calls_per_hour"] = throughput_per_hour
        out["projected_wall_days_worst_case"] = total_worst / throughput_per_hour / 24.0
        out["wall_ceiling_days"] = WALL_CEILING_DAYS
    return out


# --- runs ---------------------------------------------------------------------------------


LABEL_LIMITATIONS: Final = {
    "pilot": "Pre-freeze pilot (protocol-validation §10.1 step 4): it measures throughput, "
    "parse failures and isolation; it is not an evaluation and supports no behavioural "
    "claim.",
    "probes": "Memorisation and isolation probes (protocol-validation §6.3): verbatim "
    "answers for the owner's reading; the contamination markers are only an aid.",
    "heldout": "B4 held-out games (protocol-validation §10.3): run once per (variant, "
    "mechanism) under the frozen protocol; M1/M2 offers come from the mechanism surrogate, "
    "whose error propagates into these games.",
    "calibration": "B4-calibrated calibration games under the training mechanism "
    "Proportional only (protocol-validation §6.3); the type weights are chosen from these "
    "games and the fit-split human Proportional games, never from held-out data.",
}
PRIVACY_LIMITATION: Final = (
    "Raw call records can contain the operator's account e-mail address (injected by the "
    "CLI into the model's environment context): this run stays local under runs/ and is "
    "never published; published B4 data are parsed actions only."
)


def protocol_record() -> dict[str, Any]:
    """Path, header status and SHA-256 of the validation protocol text at run time."""
    path = data._repo_root() / data.VALIDATION_PROTOCOL_PATH  # pyright: ignore[reportPrivateUsage]
    content = path.read_bytes()
    return {
        "path": data.VALIDATION_PROTOCOL_PATH.as_posix(),
        "status": data.header_status(content.decode("utf-8")),
        "sha256": hashlib.sha256(content).hexdigest(),
        "held_out_access": "none",
    }


def _base_manifest(
    run_id: str, experiment_id: str, created_at: datetime, label: str = "pilot"
) -> dict[str, Any]:
    commit, dirty = git_source()
    timestamp = created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    limitations = [
        "LLM sampling cannot be seeded through the CLI: the games are stochastic and the "
        "run archives every prompt, raw response, retry, latency and reported model id.",
        LABEL_LIMITATIONS.get(label, LABEL_LIMITATIONS["pilot"]),
        PRIVACY_LIMITATION,
    ]
    if dirty:
        limitations.append(
            "Generated from a working tree with uncommitted or untracked changes "
            "(source.git_dirty = true); git_commit alone does not identify the code."
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "study_id": c1.STUDY_ID,
        "experiment_id": experiment_id,
        "status": "computed_unvalidated",
        "created_at": timestamp,
        "source": {
            "repository": c1.REPOSITORY,
            "git_commit": commit,
            "git_dirty": dirty,
            "generator": "phasespace.citizens.llm.runner",
            "generator_version": version("phasespace"),
        },
        "environment": environment_info(),
        "random_seed": B4_SEED,
        "artifacts": [],
        "limitations": limitations,
    }


def _metric(value: float | int | None, unit: str, definition: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "definition": definition}


def finalize_games_run(
    runs_dir: Path,
    config: RunConfig,
    outcomes: Sequence[GameOutcome],
    *,
    wall_s: float | None,
    probe_calls: int = 0,
    created_at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the immutable run from the work directory (never overwrites).

    `extra` is merged into the manifest's `citizens` block (held-out runs record the
    frozen protocol, the configuration hash, the surrogate run and the type weights).
    """
    created = created_at or datetime.now(UTC)
    work = runs_dir / WORK_DIR / config.run_id
    store = GameStore(work)
    files: list[tuple[str, str, bytes]] = [
        ("config", "config.json", (work / "config.json").read_bytes())
    ]
    records: list[dict[str, Any]] = []
    for o in outcomes:
        calls = store.calls_path(o.index)
        files.append(("llm_calls", calls.name, calls.read_bytes()))
        records.extend(_read_jsonl(calls))
    abandoned: list[dict[str, Any]] = []
    if store.abandoned.exists():
        for path in sorted(store.abandoned.iterdir()):
            abandoned.extend(_read_jsonl(path))
    if abandoned:
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in abandoned)
        files.append(("llm_calls", "abandoned.calls.jsonl", payload.encode("utf-8")))
    games_rows: list[list[Any]] = []
    rounds_rows: list[list[Any]] = []
    for o in outcomes:
        m = o.metrics() or {}
        games_rows.append(
            [
                o.index,
                o.status,
                o.reason,
                json.dumps(list(o.spec.types)),
                len(o.rounds),
                o.attempts,
                o.invalid_attempts,
                o.seconds,
                o.infrastructure_attempts,
                m.get("surplus"),
                m.get("gini"),
                m.get("depletion_round"),
                m.get("players_active_end"),
            ]
        )
        for t, r in enumerate(o.rounds):
            rounds_rows.append([o.index, t, r.pool, *r.offers, *r.returns, r.next_pool])
    files.append(("sample_table", "games.json", _games_table(config.run_id, games_rows)))
    files.append(("sample_table", "rounds.json", _rounds_table(config.run_id, rounds_rows)))
    stats = call_statistics([*records, *abandoned], wall_s)
    complete = [o for o in outcomes if o.status == "complete"]
    n_calls = len(records) + len(abandoned)
    per_game = (n_calls / len(complete)) if complete else float("nan")
    retry_fraction = stats["retries"] / stats["decisions"] if stats["decisions"] else 0.0
    projection = budget_projection(
        per_game,
        retry_fraction,
        stats.get("throughput_calls_per_hour"),
        probe_calls=probe_calls,
        pilot_calls=n_calls,
    )
    manifest = _base_manifest(
        config.run_id, f"citizens-c2-b4-{config.label}", created, config.label
    )
    protocol = protocol_record()
    game_metric_rows = [o.metrics() for o in complete]
    manifest.update(
        {
            "model": c1_model(),
            "simulator": {
                "kind": "llm_citizens",
                "variant": config.variant,
                "mechanism": config.mechanism,
                "offer_rule": "deployed",
                "n_rounds": config.n_rounds,
                "backend": config.backend,
                "environment_passed": passed_environment_names(),
                "environment_rule": "allowlist config.B4_ENV_ALLOWLIST (A28); names only",
                "retry_policy": {
                    "invalid_answer_retries": config.max_retries,
                    "infrastructure_retries": len(INFRA_BACKOFF_S),
                    "infrastructure_backoff_s": list(INFRA_BACKOFF_S),
                    "discards": "invalid answers only (A27)",
                },
                "prompt_version": config.prompt_version,
                "system_prompt": config.system_prompt,
                "rules_sha256": config.rules_sha256,
            },
            "metrics": {
                "games_complete": _metric(len(complete), "count", "games played to the end"),
                "games_discarded": _metric(
                    len(outcomes) - len(complete), "count", "games with an unrecoverable call"
                ),
                "calls": _metric(n_calls, "count", "CLI calls incl. retries and abandoned"),
                "invalid_attempts": _metric(
                    stats["invalid_attempts"], "count", "calls whose answer was not valid"
                ),
                "infrastructure_attempts": _metric(
                    stats["infrastructure_attempts"],
                    "count",
                    "calls without a usable model answer (timeout, exit, API error, model "
                    "mismatch; A27), retried, never discards",
                ),
                "latency_mean_s": _metric(
                    stats["latency_mean_s"], "s", "mean wall time of one CLI call"
                ),
                "cost_usd_total": _metric(
                    stats["cost_usd_total"], "usd", "sum of total_cost_usd reported by the CLI"
                ),
                **{
                    f"mean_{k}": _metric(
                        float(np.mean([g[k] for g in game_metric_rows if g is not None]))
                        if game_metric_rows
                        else None,
                        "coin" if k == "surplus" else "dimensionless",
                        f"mean over complete games of {k} (§8 definitions)",
                    )
                    for k in ("surplus", "gini", "depletion_round", "players_active_end")
                },
            },
            "citizens": {
                "stage": f"c2_b4_{config.label}",
                "label": config.label,
                "pre_freeze": protocol["status"] != data.FROZEN,
                "config": config.to_json(),
                "call_statistics": stats,
                "budget_projection": projection,
                "validation_protocol": protocol,
            },
        }
    )
    if extra:
        manifest["citizens"].update(extra)
    out_dir = runs_dir / config.run_id
    return write_run_directory(out_dir, manifest, files)


def c1_model() -> dict[str, Any]:
    from phasespace.citizens import surrogate

    return surrogate.citizens_model()


def _games_table(run_id: str, rows: list[list[Any]]) -> bytes:
    cols: list[tuple[str, str | None, str]] = [
        ("game", "count", "index in the pre-declared game order"),
        ("status", None, "complete or discarded"),
        ("reason", None, "why a game was discarded"),
        ("types", None, "JSON list of the four players' type keys (null: plain prompt)"),
        ("rounds", "count", "rounds played"),
        ("calls", "count", "CLI calls incl. retries"),
        ("invalid_calls", "count", "calls with an invalid answer (unparseable or out of range)"),
        ("seconds", "s", "wall time of the game"),
        ("infrastructure_calls", "count", "calls with an infrastructure error (A27)"),
        ("surplus", "coin", "mean per player-round reward (notebook definition)"),
        ("gini", "dimensionless", "notebook Gini of the players' mean rewards"),
        ("depletion_round", "count", "first round with pool < 1 (1-based), else 40"),
        ("players_active_end", "count", "players offered >= 1 coin in the last round"),
    ]
    return dumps_json(
        {
            "schema_version": 1,
            "run_id": run_id,
            "columns": [{"name": n, "unit": u, "definition": d} for n, u, d in cols],
            "rows": rows,
        }
    )


def _rounds_table(run_id: str, rows: list[list[Any]]) -> bytes:
    cols: list[tuple[str, str | None, str]] = [
        ("game", "count", "game index"),
        ("round", "count", "0-based round"),
        ("pool", "coin", "pool before allocation"),
        *[(f"offer_{i}", "coin", f"offer to player {i}") for i in range(4)],
        *[(f"return_{i}", "coin", f"whole coins returned by player {i}") for i in range(4)],
        ("next_pool", "coin", "pool after the round (game.next_pool)"),
    ]
    return dumps_json(
        {
            "schema_version": 1,
            "run_id": run_id,
            "columns": [{"name": n, "unit": u, "definition": d} for n, u, d in cols],
            "rows": rows,
        }
    )


# --- probes -------------------------------------------------------------------------------


async def run_probes(
    backend: Backend, repeats: int = PROBE_REPEATS, workers: int = WORKERS
) -> list[dict[str, Any]]:
    """§6.3 memorisation probes (a), (b), the described-study probe (c), and isolation."""
    semaphore = asyncio.Semaphore(workers)
    jobs: list[tuple[str, int, str]] = [
        (name, k, text)
        for name, text in (
            ("paper", prompts.PROBE_PAPER),
            ("rules", prompts.PROBE_RULES),
            ("described", prompts.PROBE_DESCRIBED),
        )
        for k in range(repeats)
    ]
    jobs.append(("isolation", 0, prompts.PROBE_ISOLATION))

    async def one(name: str, k: int, text: str) -> dict[str, Any]:
        started = datetime.now(UTC).isoformat()
        async with semaphore:
            result = await backend.complete(text)
        answer = result.result_text or ""
        return {
            "probe": name,
            "repeat": k,
            "started_at": started,
            "prompt": text,
            "answer": result.result_text,
            "contamination_flags": prompts.contamination_flags(answer)
            if name != "isolation"
            else [],
            **result.to_json(),
        }

    return list(await asyncio.gather(*(one(*j) for j in jobs)))


def write_probe_run(
    runs_dir: Path, run_id: str, backend_info: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest = _base_manifest(run_id, "citizens-c2-b4-probes", datetime.now(UTC), "probes")
    flagged = [r for r in records if r["contamination_flags"]]
    manifest.update(
        {
            "model": c1_model(),
            "simulator": {
                "kind": "llm_probe",
                "backend": backend_info,
                "environment_passed": passed_environment_names(),
            },
            "metrics": {
                "calls": _metric(len(records), "count", "probe calls"),
                "answers_with_flags": _metric(
                    len(flagged), "count", "answers containing a contamination marker"
                ),
            },
            "citizens": {
                "stage": "c2_b4_probes",
                "label": "pilot",
                "pre_freeze": protocol_record()["status"] != data.FROZEN,
                "validation_protocol": protocol_record(),
                "markers": list(prompts.CONTAMINATION_MARKERS),
                "consequence": "protocol-validation §6.3: if any answer states k = 22, the "
                "interpolating rule or a published result, every B4 result carries the "
                "label 'contamination risk: high' (decided by reading the answers; the "
                "markers are only an aid). Not a stop rule.",
                "call_statistics": call_statistics(
                    [{**r, "value": 0, "error": None, "attempt": 0} for r in records], None
                ),
            },
        }
    )
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")
    return write_run_directory(
        runs_dir / run_id, manifest, [("llm_calls", "probes.jsonl", payload)]
    )


# --- command line ---------------------------------------------------------------------------


def _pilot_spec(n_rounds: int) -> Callable[[int], GameSpec]:
    return lambda k: GameSpec(k, n_rounds, (None, None, None, None))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phasespace.citizens.llm.runner")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", default=ClaudeCLI().model)
    parser.add_argument("--workers", type=int, default=WORKERS)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="memorisation and isolation probes (new run)")
    probe.add_argument("--run-id", required=True)
    pilot = sub.add_parser("pilot", help="B4-plain Proportional pilot (resumable, new run)")
    pilot.add_argument("--run-id", required=True)
    pilot.add_argument("--games", type=int, default=PILOT_GAMES)
    pilot.add_argument("--rounds", type=int, default=game.N_ROUNDS)
    pilot.add_argument("--probe-calls", type=int, default=0, help="for the budget projection")
    calibrate = sub.add_parser(
        "calibrate", help="B4-calibrated calibration: 60 Proportional games (resumable)"
    )
    calibrate.add_argument("--run-id", required=True)
    calibrate.add_argument("--probe-calls", type=int, default=0, help="for the budget projection")
    args = parser.parse_args(argv)
    run_id: str = args.run_id
    validate_run_id(run_id)
    runs_dir: Path = args.runs_dir
    if (runs_dir / run_id).exists():
        raise SystemExit(f"{runs_dir / run_id} already exists; runs are never overwritten")
    backend = ClaudeCLI(model=args.model)
    if args.command == "probe":
        records = asyncio.run(run_probes(backend, workers=args.workers))
        manifest = write_probe_run(runs_dir, run_id, backend.describe(), records)
        for r in records:
            print(f"--- {r['probe']} #{r['repeat']} flags={r['contamination_flags']}")
            print(r["answer"])
        print(f"{runs_dir / run_id}: {manifest['status']}")
        return 0
    if args.command == "calibrate":
        base = calibration_specs()
        config = RunConfig(
            run_id=run_id,
            label="calibration",
            variant="calibrated",
            mechanism=PROPORTIONAL.name,
            target_games=len(base),
            n_rounds=game.N_ROUNDS,
            workers=args.workers,
            max_retries=MAX_RETRIES,
            discard_limit=discard_limit(len(base)),
            seed=B4_SEED,
            backend=backend.describe(),
        )
    else:
        config = RunConfig(
            run_id=run_id,
            label="pilot",
            variant="plain",
            mechanism=PROPORTIONAL.name,
            target_games=args.games,
            n_rounds=args.rounds,
            workers=args.workers,
            max_retries=MAX_RETRIES,
            discard_limit=discard_limit(args.games),
            seed=B4_SEED,
            backend=backend.describe(),
        )
    store = prepare_work_dir(runs_dir / WORK_DIR / run_id, config)
    specs = (
        replacement_specs(calibration_specs(), store)
        if args.command == "calibrate"
        else _pilot_spec(args.rounds)
    )
    start = time.perf_counter()
    try:
        outcomes = asyncio.run(
            run_games(
                config,
                specs,
                PROPORTIONAL,
                backend,
                store,
                log=lambda s: print(s, flush=True),
            )
        )
    except InfrastructureError as exc:
        print(f"stopped: {exc}", flush=True)
        return 3
    wall = time.perf_counter() - start
    manifest = finalize_games_run(
        runs_dir, config, outcomes, wall_s=wall, probe_calls=args.probe_calls
    )
    print(json.dumps(manifest["citizens"]["call_statistics"], indent=1))
    print(json.dumps(manifest["citizens"]["budget_projection"], indent=1))
    print(f"{runs_dir / run_id}: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
