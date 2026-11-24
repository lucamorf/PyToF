"""Shared pytest configuration: headless matplotlib and a run header."""

import os

import matplotlib

matplotlib.use("Agg")


def pytest_report_header(config):
    """Say at the top of every run how to switch on the optional tests."""
    on = bool(os.environ.get("PYTOF_TEST_NOTEBOOKS"))
    return [
        "PyToF test options:",
        f"  PYTOF_TEST_NOTEBOOKS={'1' if on else '(unset)'}: "
        + (
            "notebooks are executed (a few minutes)"
            if on
            else "notebook tests skipped; run "
            "`PYTOF_TEST_NOTEBOOKS=1 pytest` to include them"
        ),
        "  source tree: run with PYTHONPATH=<repo> so that this code, not an "
        "installed PyToF, is tested",
    ]
