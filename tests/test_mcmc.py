"""
Seeded MCMC results must not change under refactoring.

tests/golden/mcmc.json was produced by the implementation from before the
barotrope/density duplicates were merged. It depends on the random-number
stream of NumPy and emcee, so the tolerance is loose enough for round-off
but not for a changed sampler.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.cases import mcmc_summary

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "mcmc.json").read_text()
)


@pytest.fixture(scope="module")
def summary():
    return mcmc_summary()


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_mcmc_matches_golden(summary, key):
    np.testing.assert_allclose(
        summary[key], GOLDEN[key], rtol=1e-9, atol=1e-30, err_msg=key
    )


def test_both_models_reach_the_autocorrelation_check_or_stay_short(summary):
    assert len(summary["baro_autocorr"]) == 1  # 100 steps -> one check
    assert summary["dens_autocorr"] == []  # 20 steps -> none
