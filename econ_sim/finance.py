from typing import Callable

STOCK_RETURN = 0.07   # real annual
BOND_RETURN = 0.025   # real annual


def linear_glidepath(age: int) -> float:
    """
    Return stock allocation fraction in [0,1] as a function of age.
    Simple linear taper inspired by target-date funds.
    - 18: 0.9 stocks
    - 65: 0.4 stocks
    - 90: 0.2 stocks (floor)
    """
    if age < 18:
        return 0.9
    if age <= 65:
        # from 0.9 at 18 to 0.4 at 65
        return max(0.4, min(0.9, 0.9 - (age - 18) * (0.5 / (65 - 18))))
    if age <= 90:
        # from 0.4 at 65 to 0.2 at 90
        return max(0.2, 0.4 - (age - 65) * (0.2 / (90 - 65)))
    return 0.2


def portfolio_return(stock_weight: float) -> float:
    bond_weight = 1.0 - stock_weight
    return stock_weight * STOCK_RETURN + bond_weight * BOND_RETURN