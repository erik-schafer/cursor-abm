from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import numpy as np

from .agents import Agent, constant_income_profile
from .mortality import MortalityModel, default_mortality_model


@dataclass
class WorldConfig:
    population_size: int = 1000
    replacement_age: int = 0  # age for newborns on replacement (0 or 18)
    base_income: float = 35_000.0
    savings_rate: float = 0.30
    random_seed: int = 42


class World:
    def __init__(self, config: WorldConfig, mortality_model: Optional[MortalityModel] = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.mortality = mortality_model or default_mortality_model()
        self.agents: List[Agent] = []
        self._init_population()
        self.year: int = 0

    def _init_population(self) -> None:
        income_fn = constant_income_profile(self.config.base_income)
        for i in range(self.config.population_size):
            # Initialize with a spread of ages to reach quicker stationarity
            init_age = int(self.rng.integers(0, 90))
            agent = Agent(
                agent_id=i,
                age=init_age,
                wealth=0.0,
                savings_rate=self.config.savings_rate,
                income_fn=income_fn,
                mortality=self.mortality,
            )
            self.agents.append(agent)

    def step_year(self) -> None:
        dead_indices: List[int] = []
        for idx, agent in enumerate(self.agents):
            agent.step_one_year(self.rng)
            if not agent.is_alive:
                dead_indices.append(idx)
        # Replace dead agents to keep population constant
        for idx in dead_indices:
            self._replace_agent(idx)
        self.year += 1

    def _replace_agent(self, idx: int) -> None:
        old_id = self.agents[idx].agent_id
        new_agent = Agent(
            agent_id=old_id,
            age=self.config.replacement_age,
            wealth=0.0,
            savings_rate=self.config.savings_rate,
            income_fn=constant_income_profile(self.config.base_income),
            mortality=self.mortality,
        )
        self.agents[idx] = new_agent

    def get_wealth_array(self, min_age: Optional[int] = None) -> np.ndarray:
        if min_age is None:
            return np.array([a.wealth for a in self.agents if a.is_alive], dtype=float)
        return np.array([a.wealth for a in self.agents if a.is_alive and a.age >= min_age], dtype=float)