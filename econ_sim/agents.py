from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from .mortality import MortalityModel, default_mortality_model
from .finance import linear_glidepath, portfolio_return


def constant_income_profile(annual_income: float) -> Callable[[int], float]:
    def _income(age: int) -> float:
        return annual_income if age >= 18 else 0.0
    return _income


@dataclass
class Agent:
    agent_id: int
    age: int
    wealth: float
    savings_rate: float
    income_fn: Callable[[int], float]
    glidepath_fn: Callable[[int], float] = linear_glidepath
    mortality: MortalityModel = field(default_factory=default_mortality_model)
    is_alive: bool = True

    def step_one_year(self, rng: np.random.Generator) -> None:
        if not self.is_alive:
            return
        # Investment returns on current wealth
        stock_weight = self.glidepath_fn(self.age)
        r = portfolio_return(stock_weight)
        self.wealth *= (1.0 + r)

        # Income and savings (if working age)
        income = self.income_fn(self.age)
        savings = self.savings_rate * income
        self.wealth += savings

        # Age and mortality
        death_prob = self.mortality.hazard(self.age)
        self.age += 1
        if rng.random() < death_prob:
            self.is_alive = False

    def reset_as_newborn(self, starting_age: int = 0) -> None:
        self.age = starting_age
        self.wealth = 0.0
        self.is_alive = True