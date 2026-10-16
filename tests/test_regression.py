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

GOLDEN = json.loads((Path(__file__).parent / "golden" / "golden.json").read_text())

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
