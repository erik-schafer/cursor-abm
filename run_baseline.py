import argparse
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from econ_sim.world import World, WorldConfig
from econ_sim.metrics import gini


def run_sim(years: int, population: int, replacement_age: int, seed: int) -> None:
    cfg = WorldConfig(
        population_size=population,
        replacement_age=replacement_age,
        random_seed=seed,
    )
    world = World(cfg)

    for _ in range(years):
        world.step_year()

    wealth_all = world.get_wealth_array()
    wealth_adults = world.get_wealth_array(min_age=18)

    print(f"Years simulated: {years}")
    print(f"Population: {population}")
    print(f"Gini (all ages): {gini(wealth_all):.3f}")
    print(f"Gini (age >= 18): {gini(wealth_adults):.3f}")

    os.makedirs("/workspace/outputs", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(wealth_adults, bins=50, color="#4a90e2", alpha=0.85)
    plt.title("Wealth distribution (age >= 18)")
    plt.xlabel("Wealth")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path = "/workspace/outputs/baseline_wealth_hist.png"
    plt.savefig(out_path)
    print(f"Histogram saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline agent-based economic simulator")
    parser.add_argument("--years", type=int, default=400)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--replacement_age", type=int, default=0, choices=[0, 18])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sim(args.years, args.population, args.replacement_age, args.seed)