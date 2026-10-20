"""JAX citizen ports against the NumPy reference models (pmfs and closed-loop means).

Independent references: the C2 NumPy models (`phasespace.citizens.models`) and the C2
closed-loop simulator (`phasespace.citizens.simulate`). The pmf checks use random and
fitted parameters on shared random histories; the fitted-model and closed-loop checks
need the frozen fit run (`runs/citizens-c2-fit-v1`) and are skipped without it. No human
data are read.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from phasespace.citizens.fit import load_fitted
from phasespace.citizens.models import b2_condcoop as b2
from phasespace.citizens.models import b3_ewa as b3
from phasespace.citizens.models.b1_gru import B1GRUEnsemble, random_weights
from phasespace.citizens.rl import checks, twin
from phasespace.citizens.rl import citizens as rc

FIT_RUN = Path(__file__).resolve().parents[2] / "runs" / "citizens-c2-fit-v1"
HAS_FIT = (FIT_RUN / "manifest.json").exists()
needs_fit = pytest.mark.skipif(not HAS_FIT, reason="frozen fit run runs/citizens-c2-fit-v1 absent")
PMF_ATOL = 1e-6


@pytest.fixture(autouse=True)
def x64_mode() -> Iterator[None]:
    with twin.x64():
        yield


def test_b1_port_matches_reference_random_weights() -> None:
    rng = np.random.default_rng(11)
    model = B1GRUEnsemble([random_weights(rng, 0.5) for _ in range(3)])
    port = rc.B1Port.from_reference(model)
    diff, mass = checks.pmf_difference(model, port, checks.random_histories(1), seed=2)
    assert diff < PMF_ATOL and mass < 1e-12


def test_b2_port_matches_reference_random_parameters() -> None:
    theta = np.random.default_rng(12).normal(0.0, 1.0, 12)
    port = rc.B2Port(theta=twin.f64(theta))
    model = b2.B2ConditionalCooperation(theta)
    diff, mass = checks.pmf_difference(model, port, checks.random_histories(3))
    assert diff < PMF_ATOL and mass < 1e-9


@pytest.mark.parametrize("hybrid", [False, True])
def test_ewa_port_matches_reference_random_parameters(hybrid: bool) -> None:
    rng = np.random.default_rng(13 + hybrid)
    params = np.array([rng.uniform(lo, hi) for lo, hi in b3.bounds(hybrid)])
    params[3] = rng.uniform(0.5, 5.0)  # lambda in a range that gives non-degenerate choice
    model = b3.EWAModel(params, rng.dirichlet(np.ones(11)), hybrid=hybrid)
    port = rc.EWAPort.from_reference(model)
    diff, mass = checks.pmf_difference(model, port, checks.random_histories(4 + hybrid))
    assert diff < PMF_ATOL and mass < 1e-12


@needs_fit
@pytest.mark.parametrize("name", rc.TRAINABLE_MODELS)
def test_fitted_ports_match_reference(name: str) -> None:
    reference = load_fitted(FIT_RUN).sim_models()[name]
    port = rc.load_citizens(FIT_RUN)[name]
    diff, mass = checks.pmf_difference(reference, port, checks.random_histories(5), seed=6)
    assert diff < PMF_ATOL and mass < 1e-9, (name, diff, mass)


def test_sample_pmf_inverts_the_cdf() -> None:
    pmf = np.zeros((3, 201))
    pmf[0, [0, 2]] = 0.5
    pmf[1, 5] = 1.0
    pmf[2, :4] = 0.25
    u = np.array([0.49, 0.99999, 0.6])
    draws = np.asarray(rc.sample_pmf(twin.f64(pmf), twin.f64(u)))
    np.testing.assert_array_equal(draws, [0.0, 5.0, 2.0])


def test_sampled_frequencies_follow_the_pmf() -> None:
    """Inversion sampling reproduces a known pmf (5 binomial standard errors)."""
    pmf = np.zeros((1, 201))
    pmf[0, [0, 3, 7, 10]] = [0.1, 0.2, 0.3, 0.4]
    n = 200_000
    u = np.random.default_rng(7).random(n)
    draws = np.asarray(rc.sample_pmf(twin.f64(np.repeat(pmf, n, axis=0)), twin.f64(u)))
    freq = np.array([(draws == v).mean() for v in (0, 3, 7, 10)])
    np.testing.assert_allclose(freq, [0.1, 0.2, 0.3, 0.4], atol=5 * np.sqrt(0.25 / n))


@needs_fit
@pytest.mark.parametrize("name", rc.TRAINABLE_MODELS)
def test_closed_loop_means_match_reference_simulator(name: str) -> None:
    """|z| <= 4.5 for every metric under Equal, Proportional and Interpolating k = 22
    (12 comparisons per model; a false alarm has probability below 1e-4)."""
    reference = load_fitted(FIT_RUN).sim_models()[name]
    port = rc.load_citizens(FIT_RUN)[name]
    for i, mech in enumerate(("Equal", "Proportional", "Interpolating")):
        z = checks.closed_loop_z(reference, port, mech, 1500, seed=100 + i)
        assert max(z.values()) <= 4.5, (name, mech, z)
