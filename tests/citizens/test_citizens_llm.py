"""B4 LLM citizens: prompt text, strict parsing, game driver, resume (no real LLM calls).

References: a hand-written golden prompt (values of a hand-computed Proportional game);
the game rules applied directly (`mechanisms.paper_offers`-free arithmetic by hand where
small); a fake backend whose answers are a known function of the prompt.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from phasespace.citizens import game, metrics
from phasespace.citizens.llm import backend as be
from phasespace.citizens.llm import prompts, runner
from phasespace.citizens.llm.backend import CallResult
from phasespace.citizens.mechanisms import baseline_offers

GOLDEN_RULES = """\
You are one of four players in a game. The four players share a pool of coins.

How a round works:
- The pool starts at 200 coins and can never hold more than 200 coins.
- At the start of each round, coins from the pool are offered to the four players. \
You can see how many coins each player is offered. Offers can differ between players \
and from round to round.
- Each player then decides how many of the coins offered to them to return to the \
pool, and keeps the rest. You can return any whole number of coins from 0 up to the \
whole coins offered to you.
- Every coin returned to the pool grows into 1.4 coins. Coins that players keep leave \
the pool.
- The pool for the next round is: this round's pool, minus all coins offered, plus 1.4 \
times all coins returned (never more than 200).
- A player offered less than 1 coin cannot return anything that round.
- You do not know how many rounds the game will last.

Payment: your bonus is proportional to the total number of coins you keep over the game."""

# Round 1: pool 200, equal offers 50, returns (20, 30, 10, 40), sum 100 -> pool 140.
# Round 2 (Proportional): offers 140 * (.2, .3, .1, .4) = (28, 42, 14, 56); returns
# (10, 20, 0, 30), sum 60 -> pool 140 - 140 + 84 = 84.
# Round 3: offers 84 * (10, 20, 0, 30) / 60 = (14, 28, 0, 42); player 0 may return 0..14.
# Kept: 50 - 20 = 30, 28 - 10 = 18, total 48.
GOLDEN_ROUND3_T3 = (
    GOLDEN_RULES
    + """

About you: you are the kind of person who returns coins generously to keep the pool \
healthy.

The other three players are shown as B, C and D.

Previous rounds (coins):
| Round | Pool at start | Offer: You | Offer: B | Offer: C | Offer: D | Returned: You \
| Returned: B | Returned: C | Returned: D | You kept |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 200.00 | 50.00 | 50.00 | 50.00 | 50.00 | 20 | 30 | 10 | 40 | 30.00 |
| 2 | 140.00 | 28.00 | 42.00 | 14.00 | 56.00 | 10 | 20 | 0 | 30 | 18.00 |

Coins you have kept so far: 48.00

This is round 3. The pool holds 84.00 coins.
Offers this round: You 14.00, B 28.00, C 0.00, D 42.00.

You may return any whole number of coins from 0 to 14. How many of your coins do you \
return to the pool? Reply with only a JSON object of the form {"return": N}, where N is \
a whole number from 0 to 14.
"""
)

# Fractional values, written by hand: every coin amount is truncated (never rounded) to
# 2 decimals. Own offer 1.9996 -> "1.99" (largest valid return 1); 49.99975 -> "49.99";
# 0.29 -> "0.29"; 99.999 -> "99.99"; 37.7067 -> "37.70"; pool 140.0049 -> "140.00";
# kept 49.99975 - 20 = 29.99975 -> "29.99". (Values are chosen for the display, not
# produced by a mechanism.)
GOLDEN_FRACTIONAL_PLAIN = (
    GOLDEN_RULES
    + """

The other three players are shown as B, C and D.

Previous rounds (coins):
| Round | Pool at start | Offer: You | Offer: B | Offer: C | Offer: D | Returned: You \
| Returned: B | Returned: C | Returned: D | You kept |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 200.00 | 49.99 | 49.99 | 49.99 | 49.99 | 20 | 30 | 10 | 40 | 29.99 |

Coins you have kept so far: 29.99

This is round 2. The pool holds 140.00 coins.
Offers this round: You 1.99, B 0.29, C 99.99, D 37.70.

You may return any whole number of coins from 0 to 1. How many of your coins do you \
return to the pool? Reply with only a JSON object of the form {"return": N}, where N is \
a whole number from 0 to 1.
"""
)


def _golden_observation(player: int = 0) -> prompts.Observation:
    history = [
        runner.RoundRecord(200.0, [50.0] * 4, [20, 30, 10, 40], 140.0, [True] * 4),
        runner.RoundRecord(140.0, [28.0, 42.0, 14.0, 56.0], [10, 20, 0, 30], 84.0, [True] * 4),
    ]
    return runner.observation(player, 2, 84.0, np.array([14.0, 28.0, 0.0, 42.0]), history)


def test_rules_text_is_the_golden_text() -> None:
    assert prompts.RULES == GOLDEN_RULES


def test_golden_prompt() -> None:
    assert prompts.render_prompt(_golden_observation(), "T3") == GOLDEN_ROUND3_T3


def test_golden_prompt_truncates_fractional_coins() -> None:
    obs = prompts.Observation(
        round_index=1,
        pool=140.0049,
        offers=(1.9996, 0.29, 99.999, 37.7067),
        history=(
            prompts.PastRound(200.0, (49.99975, 49.99975, 49.99975, 49.99975), (20, 30, 10, 40)),
        ),
    )
    assert obs.max_return == 1
    assert prompts.render_prompt(obs) == GOLDEN_FRACTIONAL_PLAIN


@pytest.mark.parametrize(
    ("x", "shown"),
    [
        (49.99975, "49.99"),
        (1.9996, "1.99"),
        (2.0 - 1e-12, "1.99"),
        (0.29, "0.29"),  # decimal truncation of the shortest repr, not of 0.28999...
        (0.005, "0.00"),
        (14.0, "14.00"),
        (200.0, "200.00"),
        (-0.0, "0.00"),
        (np.float64(49.99975), "49.99"),
        (np.float32(0.5), "0.50"),
    ],
)
def test_coins_are_truncated_to_two_decimals(x: float, shown: str) -> None:
    assert prompts.coins(x) == shown


_OWN_OFFER = re.compile(r"Offers this round: You ([0-9]+\.[0-9]{2}),")
_RANGE = re.compile(r"You may return any whole number of coins from 0 to ([0-9]+)\. ")


@pytest.mark.parametrize("e", [49.99975, 1.9996, 1.0, 2.0 - 1e-12, 14.0, 199.999])
def test_displayed_own_offer_never_exceeds_the_allowed_maximum(e: float) -> None:
    obs = prompts.Observation(0, 200.0, (e, 10.0, 10.0, 10.0), ())
    text = prompts.render_prompt(obs)
    own = _OWN_OFFER.search(text)
    stated = _RANGE.search(text)
    assert own is not None and stated is not None
    assert int(stated.group(1)) == math.floor(e) == obs.max_return
    # The displayed offer is below the next whole number: its whole part is the maximum.
    assert float(own.group(1)) <= e
    assert math.floor(float(own.group(1))) == math.floor(e)
    assert text.endswith(f"where N is a whole number from 0 to {math.floor(e)}.\n")


def test_displayed_offers_keep_their_whole_part_on_random_values() -> None:
    rng = np.random.default_rng(20261020)
    for e in rng.uniform(0.0, 200.0, 20_000):
        shown = prompts.coins(float(e))
        assert math.floor(float(shown)) == math.floor(e)
        assert 0.0 <= float(e) - float(shown) < 0.01 + 1e-9


def test_rotation_puts_the_focal_player_first() -> None:
    obs = _golden_observation(player=2)
    assert obs.offers == (0.0, 42.0, 14.0, 28.0)
    assert obs.history[0].returns == (10, 40, 20, 30)
    assert obs.max_return == 0
    assert prompts.rotate([1, 2, 3, 4], 3) == (4, 1, 2, 3)


def test_prompts_are_neutral() -> None:
    texts = [prompts.render_prompt(_golden_observation(), t) for t in (None, *prompts.TYPE_KEYS)]
    forbidden = re.compile(
        r"\bAI\b|\bRL\b|manager|agent|deepmind|koster|clone|nature|paper|algorithm",
        re.IGNORECASE,
    )
    for text in [*texts, prompts.SYSTEM_PROMPT]:
        assert forbidden.search(text) is None, forbidden.search(text)


@pytest.mark.parametrize(
    ("text", "value", "error"),
    [
        ('{"return": 7}', 7, None),
        ('  {"return":0}\n', 0, None),
        ('{ "return" : 14 }', 14, None),
        ('{"return": 15}', None, "range"),
        ('{"return": -1}', None, "range"),
        ("7", None, "format"),
        ('{"return": 7} I keep the rest.', None, "format"),
        ('```json\n{"return": 7}\n```', None, "format"),
        ('{"return": "7"}', None, "format"),
        ('{"return": 7.0}', None, "format"),
        ('{"return": 1, "return": 2}', None, "format"),
        ('{"Return": 7}', None, "format"),
        ('{"return": 7, "why": "x"}', None, "format"),
        ("", None, "format"),
        (None, None, "no_result"),
    ],
)
def test_strict_answer_parsing(text: str | None, value: int | None, error: str | None) -> None:
    assert prompts.parse_answer(text, 14) == prompts.Parsed(value, error)


def test_cli_argv_and_response_interpretation() -> None:
    cli = be.ClaudeCLI()
    argv = cli.argv()
    assert argv[:2] == ["claude", "-p"]
    for flag, value in (
        ("--model", "claude-haiku-5-5"),
        ("--effort", "medium"),
        ("--output-format", "json"),
        ("--tools", ""),
        ("--setting-sources", ""),
        ("--max-turns", "1"),
        ("--system-prompt", prompts.SYSTEM_PROMPT),
    ):
        assert argv[argv.index(flag) + 1] == value
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    stdout = json.dumps(
        {
            "type": "result",
            "is_error": False,
            "result": '{"return": 3}',
            "total_cost_usd": 0.0001,
            "duration_ms": 900,
            "session_id": "s",
            "usage": {"output_tokens": 9},
            "modelUsage": {"claude-haiku-5-5": {}},
        }
    )
    r = be.interpret(stdout, "", 0, 1.5)
    assert (r.result_text, r.is_error, r.cost_usd, r.models) == (
        '{"return": 3}',
        False,
        0.0001,
        ["claude-haiku-5-5"],
    )
    assert be.interpret("not json", "boom", 1, 0.1).is_error
    assert be.interpret(stdout, "", 2, 0.1).is_error  # non-zero exit


def _stdout(**fields: Any) -> str:
    base: dict[str, Any] = {
        "is_error": False,
        "result": '{"return": 3}',
        "modelUsage": {"claude-haiku-5-5": {}},
    }
    return json.dumps({**base, **fields})


def test_infrastructure_errors_are_classified() -> None:
    model = "claude-haiku-5-5"
    assert be.interpret(_stdout(), "", 0, 1.0, expected_model=model).infra_error is None
    cases = {
        "nonzero_exit": be.interpret(_stdout(), "", 1, 1.0, expected_model=model),
        "unreadable_output": be.interpret("not json", "", 0, 1.0, expected_model=model),
        "api_error": be.interpret(
            _stdout(is_error=True, result="API Error: 429 rate limited"), "", 0, 1.0, model
        ),
        "no_result": be.interpret(_stdout(result=None), "", 0, 1.0, expected_model=model),
        "model_mismatch": be.interpret(
            _stdout(modelUsage={model: {}, "claude-sonnet-5-5": {}}), "", 0, 1.0, model
        ),
    }
    for kind, result in cases.items():
        assert result.infra_error == kind
    assert be.interpret(_stdout(modelUsage={}), "", 0, 1.0, model).infra_error == "model_mismatch"
    # An answer that is not valid JSON of the right form is the model's answer, not an
    # infrastructure error.
    assert be.interpret(_stdout(result="I keep 3"), "", 0, 1.0, model).infra_error is None
    assert be.cli_version_number("2.1.304 (Claude Code)") == "2.1.304"
    assert be.cli_version_number(None) is None


def test_call_environment_is_an_allowlist() -> None:
    source = {
        "PATH": "/bin",
        "HOME": "/h",
        "USER": "u",
        "LANG": "en",
        "TMPDIR": "/t",
        "CLAUDE_EFFORT": "max",
        "CLAUDECODE": "1",
        "AI_AGENT": "x",
        "ANTHROPIC_API_KEY": "secret",
        "HF_TOKEN": "secret",
    }
    env = be.call_environment(source)
    assert env == {"PATH": "/bin", "HOME": "/h", "USER": "u", "LANG": "en", "TMPDIR": "/t"}
    assert be.passed_environment_names(source) == ["HOME", "LANG", "PATH", "TMPDIR", "USER"]
    assert set(be.ENV_ALLOWLIST) == {
        "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "TERM"
    }  # fmt: skip


def test_cli_subprocess_gets_only_allowlisted_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Proc:
        returncode = 0

        async def communicate(self, data: bytes) -> tuple[bytes, bytes]:
            return _stdout().encode(), b""

    async def fake_exec(*argv: str, **kwargs: Any) -> Proc:
        captured.update(kwargs, argv=list(argv))
        return Proc()

    monkeypatch.setenv("CLAUDE_EFFORT", "max")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setattr(be.asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(be.ClaudeCLI().complete("prompt"))
    assert result.infra_error is None and result.result_text == '{"return": 3}'
    assert set(captured["env"]) <= set(be.ENV_ALLOWLIST)
    assert "PATH" in captured["env"]
    assert captured["argv"] == be.ClaudeCLI().argv()


class FakeBackend:
    """Answers a known function of the prompt: return floor(max_return * fraction)."""

    def __init__(self, fraction: float = 0.5, script: list[str] | None = None) -> None:
        self.fraction = fraction
        self.script = list(script or [])
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> CallResult:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        if self.script:
            text = self.script.pop(0)
        else:
            n = int(re.findall(r"from 0 to (\d+)", prompt)[-1])
            text = json.dumps({"return": math.floor(n * self.fraction)})
        return CallResult("{}", "", 0, 0.01, False, text, False, 0.0, ["fake"])

    def describe(self) -> dict[str, Any]:
        return {"kind": "fake", "fraction": self.fraction}


def _play(backend: FakeBackend, n_rounds: int = 40) -> tuple[runner.GameOutcome, list[Any]]:
    records: list[dict[str, Any]] = []
    spec = runner.GameSpec(0, n_rounds, (None, None, None, None))

    async def go() -> runner.GameOutcome:
        return await runner.play_game(
            spec, runner.PROPORTIONAL, backend, asyncio.Semaphore(8), records.append
        )

    return asyncio.run(go()), records


def test_game_driver_reproduces_a_game_played_by_the_rules() -> None:
    outcome, records = _play(FakeBackend(0.5))
    assert outcome.status == "complete"
    # Independent replay: Proportional (deployed) offers, each player returns floor(floor(e)/2).
    pool, prev_c, calls = game.MAX_POOL, None, 0
    rewards: list[npt.NDArray[np.float64]] = []
    pools: list[float] = []
    for t in range(40):
        e = baseline_offers(pool, prev_c, 0.0)
        c = np.array([math.floor(math.floor(x) * 0.5) if x >= 1 else 0 for x in e], float)
        calls += int(np.count_nonzero(e >= 1))
        r = outcome.rounds[t]
        np.testing.assert_array_equal(r.offers, e)
        assert r.returns == [int(v) for v in c]
        assert r.pool == pool
        pools.append(pool)
        rewards.append(e - c)
        pool, prev_c = float(game.next_pool(pool, e, c)), c
        assert r.next_pool == pool
    assert len(records) == calls == outcome.attempts
    rw, n = np.array([rewards]), np.array([40])
    m = outcome.metrics()
    assert m is not None
    assert m["surplus"] == pytest.approx(float(metrics.surplus(rw, n)[0]))
    assert m["gini"] == pytest.approx(float(metrics.reward_gini(rw, n)[0]))
    assert m["depletion_round"] == float(metrics.depletion_round(np.array([pools]), n)[0])
    # Determinism: a second run with the same fake gives the same game.
    again, _ = _play(FakeBackend(0.5))
    assert again.to_json()["rounds"] == outcome.to_json()["rounds"]


def test_excluded_players_are_not_asked() -> None:
    # Everyone keeps everything: after round 1 the pool is max(0.01, 200 - 199.999 + 0)
    # = 0.01, so every later offer is about 0.0022 < 1 and nobody is asked again.
    outcome, records = _play(FakeBackend(0.0), n_rounds=5)
    assert outcome.status == "complete"
    assert len(records) == 4  # only round 1
    assert all(not any(r.called) for r in outcome.rounds[1:])
    assert all(r.returns == [0, 0, 0, 0] for r in outcome.rounds)


def test_retry_then_success_and_unrecoverable_discard() -> None:
    fake = FakeBackend(0.5, script=["sure, 25", '{"return": 25}'])
    outcome, records = _play(fake, n_rounds=1)
    assert outcome.status == "complete"
    retried = [r for r in records if r["attempt"] == 1]
    assert len(retried) == 1
    # Deployed round-1 offer (200 - 0.001) / 4 = 49.99975, so the answer range is 0..49.
    assert prompts.retry_reminder(49) in retried[0]["prompt"]
    # The deployed round-1 offer is shown truncated, below the stated maximum's next coin.
    assert "Offers this round: You 49.99, B 49.99, C 49.99, D 49.99." in fake.prompts[0]
    assert "You may return any whole number of coins from 0 to 49. " in fake.prompts[0]
    assert outcome.invalid_attempts == 1
    bad = FakeBackend(0.5, script=["nope"] * 12)
    outcome, records = _play(bad, n_rounds=3)
    assert outcome.status == "discarded"
    assert len(records) == 12  # 4 players x (1 + 2 retries) in round 1
    assert outcome.rounds == []


class ScriptedBackend:
    """Returns scripted CallResults (then valid answers of half the maximum)."""

    def __init__(self, script: list[CallResult]) -> None:
        self.script = list(script)
        self.calls = 0

    async def complete(self, prompt: str) -> CallResult:
        self.calls += 1
        await asyncio.sleep(0)
        if self.script:
            return self.script.pop(0)
        n = int(re.findall(r"from 0 to (\d+)", prompt)[-1])
        return _answer(json.dumps({"return": n // 2}))

    def describe(self) -> dict[str, Any]:
        return {"kind": "scripted"}


def _answer(text: str | None) -> CallResult:
    return CallResult("{}", "", 0, 0.01, False, text, False, 0.0, ["claude-haiku-5-5"])


TIMEOUT = CallResult("", "", None, 120.0, True, None, True, None, infra_error="timeout")
RATE_LIMIT = CallResult(
    "{}", "", 0, 1.0, False, "API Error: 429", True, None, infra_error="api_error"
)
MISMATCH = dataclasses.replace(_answer('{"return": 3}'), infra_error="model_mismatch")
NO_WAIT = (0.0,) * 6


def _decide(backend: ScriptedBackend, records: list[dict[str, Any]]) -> runner.Decision:
    return asyncio.run(
        runner.decide(
            "Reply. from 0 to 49",
            49,
            backend,
            asyncio.Semaphore(1),
            {},
            records.append,
            infra_backoff=NO_WAIT,
        )
    )


def test_infrastructure_errors_have_their_own_retry_budget() -> None:
    # 6 infrastructure errors and 2 invalid answers, then a valid answer: the decision
    # succeeds (the invalid-answer budget of 2 retries is not used by the infra errors).
    script = [TIMEOUT, RATE_LIMIT, _answer("sure"), MISMATCH, TIMEOUT, _answer("50"),
              TIMEOUT, RATE_LIMIT, _answer('{"return": 7}')]  # fmt: skip
    records: list[dict[str, Any]] = []
    d = _decide(ScriptedBackend(script), records)
    assert d.value == 7
    classes = [r["error_class"] for r in records]
    assert classes.count("infrastructure") == 6
    assert classes.count("invalid_answer") == 2
    # The format reminder is added only after an invalid answer, never after an
    # infrastructure error alone.
    assert "could not be read" not in records[1]["prompt"]
    assert "could not be read" in records[3]["prompt"]
    stats = runner.call_statistics(records, None)
    assert (stats["infrastructure_attempts"], stats["invalid_attempts"]) == (6, 2)


def test_seven_infrastructure_errors_in_a_decision_stop_the_run() -> None:
    with pytest.raises(runner.InfrastructureError):
        _decide(ScriptedBackend([TIMEOUT] * 7), [])


def test_backoff_waits_double_from_5_to_160_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    with pytest.raises(runner.InfrastructureError):
        asyncio.run(
            runner.decide(
                "from 0 to 9",
                9,
                ScriptedBackend([TIMEOUT] * 7),
                asyncio.Semaphore(1),
                {},
                lambda r: None,
            )
        )
    # (the scripted backend's own sleep(0) calls are not backoff waits)
    assert [w for w in waits if w > 0] == [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]


def test_infrastructure_errors_never_discard_a_game(tmp_path: Path) -> None:
    # Every decision of round 1 meets 3 infrastructure errors first: no discard, and the
    # game outcome counts them apart from invalid answers.
    backend = ScriptedBackend([TIMEOUT] * 3 + [RATE_LIMIT] * 3 + [MISMATCH] * 3 + [TIMEOUT] * 3)
    outcome = asyncio.run(
        runner.play_game(runner.GameSpec(0, 2, (None, None, None, None)), runner.PROPORTIONAL,
                         backend, asyncio.Semaphore(1), lambda r: None, infra_backoff=NO_WAIT)
    )  # fmt: skip
    assert outcome.status == "complete"
    assert (outcome.infrastructure_attempts, outcome.invalid_attempts) == (12, 0)
    again = runner.GameOutcome.from_json(json.loads(json.dumps(outcome.to_json())))
    assert again.infrastructure_attempts == 12
    legacy = {k: v for k, v in outcome.to_json().items() if k != "infrastructure_attempts"}
    assert runner.GameOutcome.from_json(legacy).infrastructure_attempts == 0


def test_legacy_records_are_classified_by_error_kind() -> None:
    assert runner.error_class({"value": None, "error": "timeout"}) == "infrastructure"
    assert runner.error_class({"value": None, "error": "cli_error"}) == "infrastructure"
    assert runner.error_class({"value": None, "error": "range"}) == "invalid_answer"
    assert runner.error_class({"value": 3, "error": None}) is None


def _config(tmp_path: Path, games: int = 2) -> runner.RunConfig:
    return runner.RunConfig(
        run_id="test-run",
        label="test",
        variant="plain",
        mechanism=runner.PROPORTIONAL.name,
        target_games=games,
        n_rounds=3,
        workers=4,
        max_retries=2,
        discard_limit=runner.discard_limit(games),
        seed=runner.B4_SEED,
        backend={"kind": "fake"},
    )


class ExplodingBackend(FakeBackend):
    async def complete(self, prompt: str) -> CallResult:
        raise AssertionError("a finished game was replayed")


def test_resume_skips_finished_games_and_archives_interrupted_calls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    work = tmp_path / runner.WORK_DIR / config.run_id
    store = runner.prepare_work_dir(work, config)

    def spec(k: int) -> runner.GameSpec:
        return runner.GameSpec(k, 3, (None, None, None, None))

    first = asyncio.run(runner.run_games(config, spec, runner.PROPORTIONAL, FakeBackend(), store))
    assert [o.status for o in first] == ["complete", "complete"]
    # An interrupted attempt at game 2 leaves a partial file; a resume moves it aside.
    (work / "game-0002.calls.partial.jsonl").write_text('{"x": 1}\n')
    store.clear_interrupted(2)
    assert not (work / "game-0002.calls.partial.jsonl").exists()
    assert len(list((work / "abandoned").iterdir())) == 1
    again = asyncio.run(
        runner.run_games(config, spec, runner.PROPORTIONAL, ExplodingBackend(), store)
    )
    assert [o.to_json() for o in again] == [o.to_json() for o in first]
    with pytest.raises(ValueError, match="identical"):
        runner.prepare_work_dir(work, _config(tmp_path, games=3))
    with pytest.raises(FileExistsError):
        store.finish(first[0])


def test_discarded_games_are_replaced_up_to_the_limit(tmp_path: Path) -> None:
    # workers=2 -> one game at a time, so the scripted failures all hit game 0.
    config = dataclasses.replace(_config(tmp_path, games=2), workers=2)  # limit: 1 discard
    store = runner.prepare_work_dir(tmp_path / "w", config)
    # Game 0 fails all 12 attempts of round 1; games 1 and 2 succeed.
    fake = FakeBackend(0.5, script=["bad"] * 12)

    def spec(k: int) -> runner.GameSpec:
        return runner.GameSpec(k, 3, (None, None, None, None))

    outcomes = asyncio.run(runner.run_games(config, spec, runner.PROPORTIONAL, fake, store))
    statuses = [o.status for o in outcomes]
    assert statuses.count("complete") == 2
    assert statuses.count("discarded") == 1
    assert len(outcomes) == 3


def test_calibration_design_and_type_draws() -> None:
    assert len(runner.compositions()) == 15
    specs = runner.calibration_specs()
    assert len(specs) == 60
    counts: dict[tuple[int, int, int], int] = {}
    for s in specs:
        key = (s.types.count("T1"), s.types.count("T2"), s.types.count("T3"))
        counts[key] = counts.get(key, 0) + 1
    assert counts == dict.fromkeys(runner.compositions(), 4)
    assert [s.to_json() for s in runner.calibration_specs()] == [s.to_json() for s in specs]
    t = runner.evaluation_types([0.2, 0.5, 0.3], 7)
    assert t == runner.evaluation_types([0.2, 0.5, 0.3], 7)
    assert set(t) <= {"T1", "T2", "T3"}
    assert runner.evaluation_types([1.0, 0.0, 0.0], 3) == ("T1", "T1", "T1", "T1")
    with pytest.raises(ValueError):
        runner.evaluation_types([0.5, 0.5, 0.5], 0)


def test_budget_projection_by_hand() -> None:
    p = runner.budget_projection(150.0, 0.01, 2000.0, probe_calls=16, pilot_calls=300)
    # 60 calibration + 240 evaluation games.
    assert p["projected_calls_at_measured_rate"] == pytest.approx(16 + 300 + 300 * 150.0)
    assert p["projected_calls_worst_case"] == pytest.approx(16 + 300 + 300 * 160 * 1.01)
    assert p["within_ceiling_worst_case"] is True
    assert p["projected_wall_days_worst_case"] == pytest.approx(
        (16 + 300 + 300 * 160 * 1.01) / 2000.0 / 24.0
    )


def test_replacement_games_keep_the_design(tmp_path: Path) -> None:
    # A design of 2 games; game 0 is discarded, so game 2 must repeat game 0's seats.
    config = dataclasses.replace(_config(tmp_path, games=2), workers=2)
    store = runner.prepare_work_dir(tmp_path / "w", config)
    base = [
        runner.GameSpec(0, 3, ("T1", "T2", "T3", "T1")),
        runner.GameSpec(1, 3, ("T3", "T3", "T2", "T1")),
    ]
    spec = runner.replacement_specs(base, store)
    fake = FakeBackend(0.5, script=["bad"] * 12)
    outcomes = asyncio.run(runner.run_games(config, spec, runner.PROPORTIONAL, fake, store))
    assert [o.status for o in outcomes] == ["discarded", "complete", "complete"]
    assert outcomes[2].spec == runner.GameSpec(2, 3, ("T1", "T2", "T3", "T1"))
    with pytest.raises(ValueError, match="replaces no discarded game"):
        spec(3)


def test_backend_identity_excludes_the_callers_environment() -> None:
    info = be.ClaudeCLI().describe()
    assert "env_removed" not in info and "environment_passed" not in info
    assert info["effort_requested"] == "medium"
