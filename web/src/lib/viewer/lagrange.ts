/**
 * x coordinate of the L1 Lagrange point in the normalized barycentric synodic CR3BP frame.
 *
 * Used only to place a labeled reference marker. L1 is the root of the x-axis equilibrium
 * condition between the primaries, f(x) = x - (1-mu)/(x+mu)^2 + mu/(1-mu-x)^2, which is strictly
 * increasing on (-mu, 1-mu) and runs from -inf to +inf there, so bisection converges to it.
 */
export function lagrangeL1(mu: number): number {
  if (!(mu > 0 && mu < 0.5)) throw new RangeError("mu must lie in (0, 0.5).");
  const f = (x: number): number => x - (1 - mu) / (x + mu) ** 2 + mu / (1 - mu - x) ** 2;
  let lo = -mu + 1e-9;
  let hi = 1 - mu - 1e-9;
  for (let i = 0; i < 200 && hi - lo > 1e-15; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) < 0) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}
