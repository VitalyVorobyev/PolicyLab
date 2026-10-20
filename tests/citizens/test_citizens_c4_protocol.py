"""The C3/C4 configuration modules agree with Appendix A of the C4 protocol.

Freeze checklist item 3 of `docs/studies/model-citizens/protocol.md`: every constant of
Appendix A is rendered from `phasespace.citizens.rl.config` (C3) and
`phasespace.citizens.rl.c4_config` and compared line by line with the protocol text, so a
change on either side fails this test. Appendix B (protocol constants not yet in code)
is added here when it is transcribed into `c4_config` before the freeze.
"""

from __future__ import annotations

import re
from pathlib import Path

from phasespace.citizens.rl import c4_config
from phasespace.citizens.rl import config as c3

PROTOCOL = Path(__file__).resolve().parents[2] / c4_config.C4_PROTOCOL_PATH


def _appendix(name: str) -> list[str]:
    text = PROTOCOL.read_text(encoding="utf-8")
    section = text.split(f"## Appendix {name}", 1)[1]
    block = re.search(r"```text\n(.*?)```", section, flags=re.S)
    assert block is not None
    return [line for line in block.group(1).splitlines() if line.strip()]


def _g(x: float) -> str:
    return f"{x:g}"


def _sci(x: float) -> str:
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _render() -> list[str]:
    c4 = c4_config
    features = " | ".join(f"{m} {','.join(c4.OOD_FEATURES[m])}" for m in c4.OOD_MODELS)
    pilots = ", ".join(f"{m} {s}" for m, s in c4.PILOT_SEEDS.items())
    return [
        f"C3 PPO (chosen on B1 in C3): games/iteration={c3.PPO_N_GAMES} "
        f"rounds={c3.PPO_N_ROUNDS} epochs={c3.PPO_EPOCHS} minibatches={c3.PPO_MINIBATCHES} "
        f"lr={_sci(c3.PPO_LEARNING_RATE)} clip={_g(c3.PPO_CLIP)} gamma={_g(c3.PPO_GAMMA)} "
        f"gae_lambda={_g(c3.PPO_GAE_LAMBDA)} value_coef={_g(c3.PPO_VALUE_COEF)} "
        f"entropy_coef={_g(c3.PPO_ENTROPY_COEF)} max_grad_norm={_g(c3.PPO_MAX_GRAD_NORM)} "
        f"reward_scale={_g(c3.PPO_REWARD_SCALE)} hidden={c3.PPO_HIDDEN} "
        f"init_log_std={_g(c3.PPO_INIT_LOG_STD)} chunk={c3.PPO_CHUNK}",
        f"C3 keys: evaluation {c3.EVAL_KEY_SEED}, k* tuning {c3.TUNE_KEY_SEED} "
        f"(ln k in [{_g(c3.LN_K_MIN)},{_g(c3.LN_K_MAX)}] step {_g(c3.LN_K_STEP)}), "
        f"monitoring {c3.MONITOR_KEY_SEED} ({c3.MONITOR_GAMES} games), "
        f"sampled {c3.SAMPLED_EVAL_KEY_SEED} | k published={_g(c3.K_PUBLISHED)} | "
        f"excluded C3 training seeds {','.join(str(s) for s in c3.PILOT_TRAIN_SEEDS)}",
        f"C4 evaluation: games={c4.EVAL_GAMES} citizen key={c4.EVAL_KEY_SEED} "
        f"sampled noise key={c4.SAMPLED_NOISE_KEY_SEED} | rules: "
        f"{', '.join(c4.FIXED_RULES)}, {c4.K_PUBLISHED_RULE}, {c4.K_STAR_RULE} (one per model)",
        f"endgame rounds (0-based)={','.join(str(t) for t in c4.LAST_ROUNDS)} | "
        f"bootstrap B={c4.BOOTSTRAP_RESAMPLES} seed={c4.BOOTSTRAP_SEED} level={_g(c4.CI_LEVEL)}",
        f"input support: models {','.join(c4.OOD_MODELS)} | {features} | "
        f"grid bins={c4.OOD_GRID_BINS}",
        f"distillation: games={c4.DISTILL_GAMES} key={c4.DISTILL_KEY_SEED} "
        f"w bins={c4.DISTILL_W_BINS} tree depth={c4.DISTILL_TREE_DEPTH} "
        f"min leaf={c4.DISTILL_TREE_MIN_LEAF} features {','.join(c4.DISTILL_FEATURES)}",
        f"plateau: smooth {c4.PLATEAU_SMOOTH_POINTS} points, final {c4.PLATEAU_FINAL_POINTS} "
        f"points, tolerance {_g(c4.PLATEAU_TOL_FRACTION)} | pilots (excluded): {pilots} "
        f"(noise key {c4.PILOT_SAMPLED_NOISE_KEY_SEED})",
        f"B4 planner games={c4.B4_PLANNER_GAMES} | per-run training limit "
        f"{_g(c3.G3_HOURS_PER_RUN)} h",
    ]


def test_appendix_a_matches_the_configuration_modules() -> None:
    assert _render() == _appendix("A")


def test_appendix_b_lists_protocol_constants() -> None:
    # Appendix B holds the constants still to be transcribed into c4_config (P8); once
    # they are, this test renders them like Appendix A.
    assert _appendix("B")
