"""Generic Monte Carlo statistics: confidence intervals, sample-size
calculations, and a Welford streaming accumulator.

Stdlib-only, deliberately **not** a dependency on the cross-filesystem
``mmo-utils`` package (``~/Desktop/fac/2026/mmo/mmo-utils``): per
``docs/shared-ml-package.md``, that package has documented gaps (a
``float``-only ``Accumulator``, two parallel sampling models, no CI for
custom accumulators, no variance reduction) and pulling in a dependency
outside this repo's filesystem tree is premature for Phase 2. Names below
mirror ``mmo-utils`` so a later swap -- or contributing these fixes upstream
-- is mechanical. Revisit at Phase 4 (shared ML package extraction).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import NormalDist
from typing import NamedTuple

_NORMAL = NormalDist()


def _z_score(alpha: float) -> float:
    """z_{alpha/2}: the normal quantile such that P(|Z| <= z) = 1 - alpha.

    Uses ``statistics.NormalDist.inv_cdf`` (exact, stdlib) rather than a
    rational approximation.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return _NORMAL.inv_cdf(1 - alpha / 2)


class ConfidenceInterval(NamedTuple):
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True, slots=True)
class MCResult:
    """Mean/variance estimate from ``n`` i.i.d. samples."""

    mean: float
    variance: float
    n: int

    def confidence_interval(self, alpha: float = 0.05) -> ConfidenceInterval:
        if self.n < 2:
            raise ValueError("need at least 2 samples for a confidence interval")
        z = _z_score(alpha)
        half_width = z * math.sqrt(self.variance / self.n)
        return ConfidenceInterval(self.mean - half_width, self.mean + half_width)


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion.

    Used for every proportion in this package -- never the normal
    approximation. Wilson stays inside [0, 1] and has much better coverage
    than the normal approximation at small ``n`` or extreme ``p``, both of
    which occur here (rare vertices, thin per-cell VP-trajectory counts).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = _z_score(alpha)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return ConfidenceInterval((center - spread) / denom, (center + spread) / denom)


def sample_size_clt(eps: float, delta: float) -> int:
    """Worst-case (p=0.5) sample size so a two-sided (1-delta) CI on a
    [0, 1]-bounded mean has half-width <= eps, via the normal approximation.

    n >= (z_{delta/2} / (2 eps))^2
    """
    z = _z_score(delta)
    return math.ceil((z / (2 * eps)) ** 2)


def sample_size_hoeffding(eps: float, delta: float) -> int:
    """Distribution-free sample size (Hoeffding's inequality) for a
    [0, 1]-bounded variable, no normality assumption -- more conservative
    than ``sample_size_clt``.

    n >= log(2/delta) / (2 eps^2)
    """
    return math.ceil(math.log(2 / delta) / (2 * eps**2))


def two_proportion_sample_size(
    p: float, delta: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Per-arm sample size to detect a difference of ``delta`` between two
    proportions both near ``p``, at significance ``alpha`` and the given
    ``power`` (normal approximation, equal-n two-sample test).
    """
    z_alpha = _z_score(alpha)
    z_power = _NORMAL.inv_cdf(power)
    variance_term = 2 * p * (1 - p)
    return math.ceil(((z_alpha + z_power) ** 2 * variance_term) / delta**2)


class TwoProportionTest(NamedTuple):
    z: float
    p_value: float


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> TwoProportionTest:
    """Two-sided pooled z-test for a difference between two proportions
    (``k1`` successes of ``n1`` vs. ``k2`` of ``n2``)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1 and n2 must be positive")
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return TwoProportionTest(z=0.0, p_value=1.0)
    z = (p1 - p2) / se
    p_value = 2 * (1 - _NORMAL.cdf(abs(z)))
    return TwoProportionTest(z=z, p_value=p_value)


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg false-discovery-rate control across many pairwise
    comparisons (e.g. ranking 54 placement vertices). Returns a same-length,
    same-order list of booleans: True where the null is rejected at FDR
    level ``q``.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    threshold_rank = 0
    for rank, i in enumerate(order, start=1):
        if p_values[i] <= (rank / n) * q:
            threshold_rank = rank
    cutoff = p_values[order[threshold_rank - 1]] if threshold_rank else -1.0
    return [p <= cutoff for p in p_values]


@dataclass(slots=True)
class Accumulator[S]:
    """Welford streaming mean/variance accumulator over samples of any type
    ``S``, via a ``value`` projection to ``float``.

    Fixes the ``mmo-utils`` gap noted in ``docs/shared-ml-package.md``: its
    ``Accumulator`` only accepts raw floats, forcing callers to pre-extract
    scalars everywhere. Here the projection is supplied once, at
    construction, and every ``update`` call takes the raw sample (a
    ``GameRecord``, a per-vertex outcome, whatever).
    """

    value: Callable[[S], float]
    count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def update(self, sample: S) -> None:
        x = self.value(sample)
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def update_all(self, samples: Iterable[S]) -> None:
        for sample in samples:
            self.update(sample)

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ZeroDivisionError("no samples accumulated")
        return self._mean

    @property
    def variance(self) -> float:
        """Sample variance, Bessel-corrected (ddof=1)."""
        if self.count < 2:
            raise ZeroDivisionError("need at least 2 samples for variance")
        return self._m2 / (self.count - 1)

    def result(self) -> MCResult:
        return MCResult(mean=self.mean, variance=self.variance, n=self.count)
