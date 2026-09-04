"""
NYAYANTRA — Mathematical & Decision Economics Unit Tests
=====================================================
Verifies:
- Bayesian Expected Value formula calculations
- Break-even probability thresholds (tau*)
- Micro-dispute boundary conditions
- Numerical stability across extreme monetary ranges
- Calibrated probability bounds [0.0, 1.0]
"""

import pytest
import numpy as np
from src.engine import DecisionEngine
import config


@pytest.fixture
def engine():
    return DecisionEngine()


def test_expected_value_formula(engine):
    """Verifies exact arithmetic of E[EV] = (P_win * Amount) - ((1 - P_win) * Fee)."""
    # Amount = INR 4,000, Fee = INR 500, P_win = 0.80
    # E[EV] = (0.80 * 4000) - (0.20 * 500) = 3200 - 100 = INR 3100.00
    ev, tau_star, is_pos = engine.calculate_expected_value(
        win_probability=0.80,
        dispute_amount_inr=4000.0,
        arbitration_fee_inr=500.0
    )
    assert ev == 3100.00
    assert tau_star == round(500.0 / (4000.0 + 500.0), 4)
    assert is_pos is True


def test_break_even_probability_boundary(engine):
    """Verifies break-even threshold tau* = Fee / (Amount + Fee)."""
    # For Amount = INR 500, Fee = INR 500 -> tau* = 500 / 1000 = 0.50
    ev, tau_star, is_pos = engine.calculate_expected_value(
        win_probability=0.50,
        dispute_amount_inr=500.0,
        arbitration_fee_inr=500.0
    )
    assert tau_star == 0.50
    assert ev == 0.00
    assert is_pos is False  # At exact 0 EV, is_pos is False

    # Just above break-even: P = 0.501
    ev_above, _, is_pos_above = engine.calculate_expected_value(
        win_probability=0.501,
        dispute_amount_inr=500.0,
        arbitration_fee_inr=500.0
    )
    assert ev_above > 0.0
    assert is_pos_above is True


def test_micro_dispute_negative_ev_boundary(engine):
    """
    Verifies that micro-disputes (Amount < Fee) require high win probabilities
    to break even, and correctly evaluate to negative EV when P_win is moderate.
    """
    # Amount = INR 150, Fee = INR 500
    # Break-even tau* = 500 / 650 = 76.92%
    # With P_win = 60%, E[EV] = (0.60 * 150) - (0.40 * 500) = 90 - 200 = -INR 110.00
    ev, tau_star, is_pos = engine.calculate_expected_value(
        win_probability=0.60,
        dispute_amount_inr=150.0,
        arbitration_fee_inr=500.0
    )
    assert ev == -110.00
    assert round(tau_star, 4) == 0.7692
    assert is_pos is False


def test_numerical_stability_extreme_ranges(engine):
    """Verifies that mathematical calculations do not overflow, underflow, or produce NaNs on extreme inputs."""
    # 1. Zero amount
    ev_zero, tau_zero, is_pos_zero = engine.calculate_expected_value(0.99, 0.0, 500.0)
    assert np.isfinite(ev_zero)
    assert not np.isnan(ev_zero)

    # 2. Very large enterprise dispute (INR 10,000,000)
    ev_large, tau_large, is_pos_large = engine.calculate_expected_value(0.50, 10_000_000.0, 500.0)
    assert np.isfinite(ev_large)
    assert ev_large > 4_000_000.0
    assert tau_large < 0.001

    # 3. Boundary probabilities (0.0 and 1.0)
    ev_p0, _, is_pos_p0 = engine.calculate_expected_value(0.0, 5000.0, 500.0)
    assert ev_p0 == -500.00
    assert is_pos_p0 is False

    ev_p1, _, is_pos_p1 = engine.calculate_expected_value(1.0, 5000.0, 500.0)
    assert ev_p1 == 5000.00
    assert is_pos_p1 is True


def test_probability_bounds_enforcement(engine):
    """Verifies that probabilities outside [0, 1] are safely clipped to [0, 1]."""
    # Negative probability clipped to 0.0
    ev_neg, _, _ = engine.calculate_expected_value(-0.5, 1000.0, 500.0)
    assert ev_neg == -500.00

    # Over-1.0 probability clipped to 1.0
    ev_over, _, _ = engine.calculate_expected_value(1.5, 1000.0, 500.0)
    assert ev_over == 1000.00
