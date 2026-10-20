"""The L-BFGS-B call shared by the maximum-likelihood fits (B2, B3, B23).

scipy.optimize.minimize(method="L-BFGS-B") with the options of `config` (an
implementation choice: the protocol names the method only); gradients by scipy's
default 2-point finite differences, which respect the bounds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize

from phasespace.citizens import config

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MinimizeResult:
    x: FloatArray
    success: bool
    iterations: int
    message: str


def lbfgsb(
    fun: Callable[[FloatArray], float],
    x0: FloatArray,
    bounds: Sequence[tuple[float, float]] | None = None,
) -> MinimizeResult:
    box = (
        None
        if bounds is None
        else optimize.Bounds(np.array([lo for lo, _ in bounds]), np.array([hi for _, hi in bounds]))
    )

    def objective(x: FloatArray) -> float:
        return fun(np.asarray(x, dtype=np.float64))

    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=box,
        options={
            "maxiter": config.LBFGSB_MAXITER,
            "maxfun": config.LBFGSB_MAXFUN,
            "ftol": config.LBFGSB_FTOL,
            "gtol": config.LBFGSB_GTOL,
        },
    )
    return MinimizeResult(
        x=np.asarray(result.x, dtype=np.float64),
        success=bool(result.success),
        iterations=int(result.nit),
        message=str(result.message),
    )
