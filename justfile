# Show available project commands.
[default]
help:
    @just --list

# Run Python tests.
test:
    uv run pytest

# Run Python lint checks.
lint:
    uv run ruff check .

# Verify Python formatting without modifying files.
format-check:
    uv run ruff format --check .

# Format Python code.
format:
    uv run ruff check --fix-only .
    uv run ruff format .

# Run strict Python type checking.
typecheck:
    uv run basedpyright

# Run all non-mutating Python validation.
check: lint format-check typecheck

# Run the complete Python validation suite.
verify-python: check test

# Run all validation: Python checks and tests, then the web lint, typecheck, tests and build.
verify: verify-python web-verify

# Export the M1 reference run (Earth-Moon L1 Lyapunov orbit) to a new directory under runs/.
gravity-reference run_id="gravity-l1-lyapunov-reference-v2":
    uv run python -m phasespace.gravity.export --runs-dir runs --run-id {{run_id}}

# M3 (S1): dense 2-D free-return feasibility map (0.5 deg x 2 m/s grid, 90 720 propagations) as a new run under runs/.
gravity-m3-map run_id="gravity-free-return-map-2d-grid-v1" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} map --design grid2d --run-id {{run_id}}

# M3 (S4): 3-D scrambled Sobol free-return map (2^17 samples, explicit seed) as a new run under runs/.
gravity-m3-sobol run_id="gravity-free-return-map-3d-sobol17-v1" seed="20261018" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} map --design sobol3d --seed {{seed}} --log2-samples 17 --run-id {{run_id}}

# M3 (S2+S3): refine, validate and export free-return candidates from a 2-D map run (new runs under runs/).
# The M3 cap is 1000 propagations per start (design proposed 200); the search run records it.
gravity-m3-candidates map_run="gravity-free-return-map-2d-grid-v1" version="v1" cap="1000" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} candidates --map-run {{map_run}} --run-prefix gravity-free-return --version {{version}} --max-propagations-per-start {{cap}}

# M3: fine delta-v probe (theta = 229 and 251 deg, 0.02 m/s, 7002 propagations) resolving feasible-band widths.
gravity-m3-probe run_id="gravity-free-return-dv-probe-v1" workers="10":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs --workers {{workers}} map --design dvprobe --run-id {{run_id}}

# M3: derive the website's compact, lossless outcome grid from a 2-D map run (new run under runs/).
gravity-m3-web-map map_run="gravity-free-return-map-2d-grid-v1" run_id="gravity-free-return-map-2d-grid-v1-web":
    uv run python -m phasespace.gravity.free_return_export --runs-dir runs web-map --map-run {{map_run}} --run-id {{run_id}}

# M4 configuration: print the config hash and key constants (no computation; --json for everything).
gravity-m4-config *args:
    uv run python -m phasespace.gravity.m4 config {{args}}

# M4 smoke run (pre-freeze, never comparative): every method at <= 2^8 propagations (seed 9001) plus cost probes.
gravity-m4-smoke run_id="gravity-m4-smoke-v1" budget="256" workers="1" probe_draws="8":
    uv run python -m phasespace.gravity.m4 --runs-dir runs smoke --run-id {{run_id}} --budget {{budget}} --workers {{workers}} --probe-draws {{probe_draws}}

# M4 frozen-protocol stages: each refuses unless protocol.md says `frozen` and --confirm-frozen is passed.
# Tuning grid (seeds 9101-9102; every grid entry, plus D and CC once): just gravity-m4-tune --confirm-frozen
gravity-m4-tune *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs stage --stage tune {{args}}

# Pilot (seeds 9201-9203) with the tuned configurations: just gravity-m4-pilot --configs MS=0,MBH=1,DE=0,T=2,T2=0 --confirm-frozen
gravity-m4-pilot *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs stage --stage pilot {{args}}

# Main comparison (seeds 1001-1010): just gravity-m4-main --configs ... --confirm-frozen
gravity-m4-main *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs stage --stage main {{args}}

# One M4 run: just gravity-m4-run --stage main --method T --config-index 1 --seed 1011 --confirm-frozen
gravity-m4-run *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs search {{args}}

# M4 coverage ground truth (protocol 5.3 a-c, K1 check): just gravity-m4-groundtruth --confirm-frozen
gravity-m4-groundtruth *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs groundtruth {{args}}

# M4 oracle-seeded continuation reference CC*: just gravity-m4-ccstar --confirm-frozen
gravity-m4-ccstar *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs ccstar {{args}}

# M4 re-check + execution-error tolerance of a run under one injection set and its paired Gates sets
# (T_E | H_E_nominal | H_E_wide): just gravity-m4-robustness <run_id> T_E --confirm-frozen
gravity-m4-robustness run errors *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs robustness --run-id {{run}} --errors {{errors}} {{args}}

# M4 step 4: calibrate B on the D tuning catalogs (T_E, primary Gates set; protocol 9.1), reading stored runs.
# just gravity-m4-calibrate --confirm-frozen   (writes gravity-m4-calibration-v1)
gravity-m4-calibrate *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs calibrate {{args}}

# M4 step 5: configuration selection (protocol 6) at the calibrated B; prints the --configs value.
# just gravity-m4-select --confirm-frozen   (writes gravity-m4-selection-v1)
gravity-m4-select *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs select {{args}}

# M4 step 6: after the pilot and its T_E robustness, the seed count n, K3 and the 7.4 re-projection.
# just gravity-m4-pilot-analysis --confirm-frozen   (writes gravity-m4-pilot-analysis-v1)
gravity-m4-pilot-analysis *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs pilot-analysis {{args}}

# M4 step 8: the final analysis (5.3 d, 5.6 LSODA sample, 10-12) as a new run.
# just gravity-m4-analysis --confirm-frozen --workers 10   (writes gravity-m4-analysis-v1)
gravity-m4-analysis *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs analysis {{args}}

# Install the website's pinned dependencies (npm, from web/package-lock.json).
web-install:
    cd web && npm ci

# Start the website dev server.
web-dev:
    cd web && npm run dev

# Lint, typecheck, test and build the website.
web-verify:
    cd web && npm run verify

# Copy a run from runs/ into web/public/research/ after verifying its checksums (never overwrites).
web-publish-run run destination:
    node web/scripts/publish-run.mjs runs/{{run}} {{destination}}

# M4 public demo run (one seed's catalogs and uncorrected error fans; reads stored runs, re-propagates the shown elites).
# just gravity-m4-demo-export --confirm-frozen --workers 10   (writes gravity-m4-demo-atlas-v1)
gravity-m4-demo-export *args:
    uv run python -m phasespace.gravity.m4 --runs-dir runs demo-export {{args}}

# Model Citizens C1: download the Koster et al. (2025) CSV once (size + MD5 verified) and parse it into data/processed/.
citizens-data:
    uv run python -m phasespace.citizens.data download
    uv run python -m phasespace.citizens.data parse

# Model Citizens C1 data audit (identity checks, published z-values) as a new run under runs/ (never overwrites).
citizens-audit run_id="citizens-c1-data-audit-v1":
    uv run python -m phasespace.citizens.export --runs-dir runs --run-id {{run_id}}

# Model Citizens C2 (models)
# Fit M0, P0, B1 (4 members), B2, B3 and B23 on the training fit split (protocol 6) as a new run under runs/.
citizens-c2-fit run_id="citizens-c2-fit-v1-dev" *args:
    uv run python -m phasespace.citizens.fit --runs-dir runs fit --run-id {{run_id}} {{args}}

# Pre-freeze smoke (training data only): fit every model, dry-run the criterion on the training regimes, time S=1000 simulations.
citizens-c2-smoke run_id="citizens-c2-smoke-v1-dev" *args:
    uv run python -m phasespace.citizens.fit --runs-dir runs smoke --run-id {{run_id}} {{args}}

# Print the C2 configuration hash (config_sha256 of phasespace.citizens.config).
citizens-c2-config:
    uv run python -m phasespace.citizens.config

# M1/M2 mechanism surrogates on BC games only (protocol 6.4; deterministic). After the freeze, re-run and
# require bit-identical parameters: just citizens-c2-surrogates citizens-c2-surrogates-v1 --reproduce runs/citizens-c2-surrogates-pre1
citizens-c2-surrogates run_id="citizens-c2-surrogates-v1" *args:
    uv run python -m phasespace.citizens.surrogate --runs-dir runs fit --run-id {{run_id}} {{args}}

# B4 memorisation and isolation probes (16 calls; protocol 6.3).
citizens-b4-probe run_id="citizens-c2-b4-probes-v1":
    uv run python -m phasespace.citizens.llm.runner probe --run-id {{run_id}}

# B4-plain Proportional pilot (resumable; protocol 10.1 step 4): just citizens-b4-pilot citizens-c2-b4-pilot-v3 --games 2
citizens-b4-pilot run_id="citizens-c2-b4-pilot-v3" *args:
    uv run python -m phasespace.citizens.llm.runner pilot --run-id {{run_id}} {{args}}

# B4-calibrated calibration: 60 Proportional games over 4 x 15 type compositions (resumable: re-run the same command).
citizens-b4-calibrate run_id="citizens-c2-b4-calibration-v2" *args:
    uv run python -m phasespace.citizens.llm.runner calibrate --run-id {{run_id}} {{args}}

# B4-calibrated type weights from a calibration run (training data only; prints, writes nothing).
citizens-b4-weights calibration_run="runs/citizens-c2-b4-calibration-v2":
    uv run python -m phasespace.citizens.heldout b4-weights --calibration-run {{calibration_run}}

# Model Citizens post-freeze stages: each refuses unless protocol-validation.md says `frozen` (committed),
# --confirm-frozen is passed and the working tree is clean.
# G1b first (held-out z-values, pool anomaly rows): just citizens-g1b --confirm-frozen   (writes citizens-c1-g1b-v1)
citizens-g1b *args:
    uv run python -m phasespace.citizens.heldout g1b {{args}}

# The frozen held-out evaluation (surrogate acceptance, criterion, G2):
# just citizens-c2-evaluate --confirm-frozen --fitted-run runs/citizens-c2-fit-v1 --surrogate-run runs/citizens-c2-surrogates-v1 --g1b-run runs/citizens-c1-g1b-v1
citizens-c2-evaluate *args:
    uv run python -m phasespace.citizens.heldout evaluate {{args}}

# One B4 held-out run (40 games, resumable, long): just citizens-b4-heldout --confirm-frozen --variant calibrated
#   --mechanism M1 --surrogate-run runs/citizens-c2-surrogates-v1 --calibration-run runs/citizens-c2-b4-calibration-v2 --run-id citizens-c2-b4-calibrated-m1-v1
citizens-b4-heldout *args:
    uv run python -m phasespace.citizens.heldout b4 {{args}}

# B4 criterion (a) and must-beat-marginal against the evaluation run (A24): just citizens-b4-evaluate --confirm-frozen
#   --evaluation-run runs/citizens-c2-heldout-eval-v1 --b4-run runs/citizens-c2-b4-plain-interpolating-v1 --b4-run ...
citizens-b4-evaluate *args:
    uv run python -m phasespace.citizens.heldout b4-evaluate {{args}}

# Model Citizens C3 (RL planner pipeline; pre-protocol pilot). Reads only the frozen fit run
# (default runs/citizens-c2-fit-v1; another one with --fit-run <dir>).
# Baselines and k* re-tuning against B1, B2, B3, B23, plus the JAX-port checks: just citizens-c3-baselines citizens-c3-baselines-v1
citizens-c3-baselines run_id *args:
    uv run python -m phasespace.citizens.rl.stage --runs-dir runs baselines --run-id {{run_id}} {{args}}

# One PPO planner: just citizens-c3-train citizens-c3-train-b1-s1-v1 B1 1 600 [--set learning_rate=0.001 ...]
citizens-c3-train run_id model seed iterations *args:
    uv run python -m phasespace.citizens.rl.stage --runs-dir runs train --run-id {{run_id}} --model {{model}} --seed {{seed}} --iterations {{iterations}} {{args}}

# G3 (planner vs Interpolating k = 22 and k* against B1): just citizens-c3-g3 <run_id> runs/<baselines> runs/<train-1> runs/<train-2> ...
citizens-c3-g3 run_id baselines_run +train_runs:
    uv run python -m phasespace.citizens.rl.stage --runs-dir runs g3 --run-id {{run_id}} --baselines-run {{baselines_run}} --train-run {{train_runs}}

# C4 PPO feasibility pilot (own model, monitoring games only): just citizens-c4-pilot B2 901
citizens-c4-pilot model seed *args:
    uv run python -m phasespace.citizens.rl.c4 --runs-dir runs pilot --model {{model}} --seed {{seed}} --run-id citizens-c4-pilot-$(echo {{model}} | tr A-Z a-z)-s{{seed}} {{args}}

# One C4 planner (C3 hyperparameters, labelled c4): just citizens-c4-train <run_id> B2 11 <iterations> [--no-retention]
citizens-c4-train run_id model seed iterations *args:
    uv run python -m phasespace.citizens.rl.stage --runs-dir runs train --milestone c4 --run-id {{run_id}} --model {{model}} --seed {{seed}} --iterations {{iterations}} {{args}}

# C4 evidence stages; each refuses to run before the C4 protocol is frozen (--confirm-frozen).
# Cross-play matrix: just citizens-c4-crossplay <run_id> --confirm-frozen --train-run runs/<a> runs/<b> ...
citizens-c4-crossplay run_id *args:
    uv run python -m phasespace.citizens.rl.c4 --runs-dir runs crossplay --run-id {{run_id}} {{args}}

# Distillation of each planner on its own model: just citizens-c4-distill <run_id> --confirm-frozen --train-run ...
citizens-c4-distill run_id *args:
    uv run python -m phasespace.citizens.rl.c4 --runs-dir runs distill --run-id {{run_id}} {{args}}

# Planner versus B4 (LLM calls; resumable): just citizens-c4-b4 <run_id> --confirm-frozen --train-run runs/<a> --variant calibrated --calibration-run runs/citizens-c2-b4-calibration-v2
citizens-c4-b4 run_id *args:
    uv run python -m phasespace.citizens.rl.c4 --runs-dir runs b4 --run-id {{run_id}} {{args}}
