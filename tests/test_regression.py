"""
Golden-master test: today's numbers must not change under refactoring.

This proves *stability*, not *correctness* (see test_analytical.py and
test_literature.py for that). If a physics change is intended, regenerate with
`python -m tests.make_golden` and review the diff of golden.json.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.cases import GOLDEN_CASES, summarize

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "golden.json").read_text()
)

# Pure floating-point algebra; allow only platform-level round-off.
RTOL = 1e-9


def test_golden_file_covers_all_cases():
    assert set(GOLDEN) == set(GOLDEN_CASES)


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_matches_golden(name):
    got = summarize(GOLDEN_CASES[name]())
    for key, expected in GOLDEN[name].items():
        np.testing.assert_allclose(
            got[key],
            expected,
            rtol=RTOL,
            atol=1e-30,
            err_msg=f"{name}: {key}",
        )


def test_reset_undoes_simpson_fallback():
    """A density jump makes PyToF fall back to the trapezoid rule; reset()
    must undo that so that the next model does not depend on the last one."""
    import numpy as np

    from PyToF import ClassToF
    from tests.cases import R_JUPITER, jupiter_polytrope

    def jump(P, param=None):
        return np.sqrt(P) * (1 + 1.0 * (P > 0.3 * np.max(P)))

    X = ClassToF.ToF(N=256, R_ref=R_JUPITER)
    X.set_barotrope(jump)
    X.relax_to_barotrope()
    assert not X.simpson_in_use

    X.reset()
    assert X.simpson_in_use
    X.set_barotrope(lambda P, param=None: np.sqrt(P))
    X.relax_to_barotrope()
    fresh = jupiter_polytrope(4, 256)
    assert X.simpson_in_use
    np.testing.assert_allclose(X.Js, fresh.Js, rtol=1e-12)


def test_invalid_input_raises_errors_not_assertions():
    import numpy as np

    from PyToF.AlgoToF import Algorithm
    from PyToF.messages import PyToFError, RenormalisationError

    with pytest.raises(ValueError, match="m_rot is negative"):
        Algorithm(np.linspace(0.1, 1, 8), np.ones(8), -1.0)
    with pytest.raises(ValueError, match="same shape"):
        Algorithm(np.linspace(0.1, 1, 8), np.ones(7), 0.1)
    with pytest.raises(NotImplementedError, match="n_bin"):
        Algorithm(np.linspace(0.1, 1, 8), np.ones(8), 0.1, n_bin=[1, 2])

    # The run-time errors share one base class:
    assert issubclass(RenormalisationError, PyToFError)
