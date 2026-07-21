"""Tests for M5 metamaterials: the Miura should come out auxetic, recoverable."""

import numpy as np

from foldforge.materials import (
    poisson_curve, deployment_ratio, fit_sector_angle,
)


def test_miura_is_auxetic():
    ths = np.radians(np.linspace(20, 75, 12))
    for alpha_deg in (40, 55, 70):
        nu = poisson_curve(np.radians(alpha_deg), ths)
        assert np.all(nu < 0)                      # negative Poisson everywhere


def test_deployment_ratio_in_bounds():
    d = deployment_ratio(np.radians(60), np.radians(60))
    assert 0 < d < 1                               # folds to a smaller footprint


def test_inverse_design_recovers_sector_angle():
    ths = np.radians(np.linspace(20, 75, 14))
    true = np.radians(52)
    tgt = poisson_curve(true, ths)
    rec = fit_sector_angle(ths, tgt)
    assert abs(np.degrees(rec) - 52) < 1.0
