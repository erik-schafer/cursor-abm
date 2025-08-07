import math
from typing import List, Callable


class MortalityModel:
    """
    Discrete-time mortality model with an age-specific hazard h(age) in [0,1).

    Calibrates a smooth sigmoid hazard curve so that expected lifetime at birth
    is approximately the provided target life expectancy (in years).
    """

    def __init__(self, target_life_expectancy_years: float = 78.0, max_age: int = 120, seed: int | None = None) -> None:
        self.target_le = target_life_expectancy_years
        self.max_age = max_age
        # Fixed shape parameters; we will solve for scale to match target LE.
        self.base_hazard = 0.0002  # infant/young baseline annual hazard
        self.midpoint_age = 70.0   # age where hazard accelerates
        self.steepness = 0.12      # logistic slope; higher = steeper increase with age
        self._scale = self._calibrate_scale()

    def hazard(self, age: int) -> float:
        # Logistic rise plus base, bounded to < 1
        z = 1.0 / (1.0 + math.exp(-self.steepness * (float(age) - self.midpoint_age)))
        h = self.base_hazard + self._scale * z
        # Cap to avoid probabilities >= 1
        return min(max(h, 0.0), 0.95)

    def survival_curve(self) -> List[float]:
        """Return S(age) for integer ages 0..max_age inclusive."""
        S = [1.0]
        for a in range(0, self.max_age):
            h = self.hazard(a)
            S.append(S[-1] * (1.0 - h))
        return S

    def expected_life_years(self) -> float:
        # E[L] for discrete-time process ~= sum_{a=0}^{max_age-1} S(a)
        S = self.survival_curve()
        return sum(S[:-1])

    def _calibrate_scale(self) -> float:
        """
        Binary search for scale so that expected life ~= target.
        scale multiplies the logistic component (0..1) added to base.
        """
        # Bracket reasonable scales
        lo, hi = 0.001, 0.8
        target = self.target_le

        def set_scale(s: float) -> float:
            self._scale = s
            return self.expected_life_years()

        # Expand bounds if needed
        for _ in range(20):
            le_lo = set_scale(lo)
            le_hi = set_scale(hi)
            if le_lo >= target:
                hi = lo
                lo = max(lo * 0.5, 1e-6)
            elif le_hi <= target:
                lo = hi
                hi = min(hi * 2.0, 2.0)
            else:
                break
        # Binary search
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            le_mid = set_scale(mid)
            if le_mid > target:
                # too long life -> increase hazard scale
                lo = mid
            else:
                hi = mid
        self._scale = 0.5 * (lo + hi)
        return self._scale


def default_mortality_model() -> MortalityModel:
    return MortalityModel(target_life_expectancy_years=78.0)