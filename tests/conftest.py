"""Shared pytest configuration: headless matplotlib, skip slow tests by default."""

import matplotlib
import pytest

matplotlib.use("Agg")


def pytest_collection_modifyitems(config, items):
    # `pytest` skips slow tests unless the user selects them with -m.
    if config.getoption("-m"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with `pytest -m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
