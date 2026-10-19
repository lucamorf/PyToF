"""
Execute the tutorial notebooks and the quick accuracy notebook.

Slow (a few minutes) and needs jupyter, so it is skipped unless you ask for it:

    PYTOF_TEST_NOTEBOOKS=1 pytest tests/test_notebooks.py

(accuracy/accuracy_full.ipynb takes tens of minutes and is not part of this.)
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "tutorial").glob("0*.ipynb")) + [
    ROOT / "accuracy" / "accuracy_quick.ipynb"
]

pytestmark = pytest.mark.skipif(
    not os.environ.get("PYTOF_TEST_NOTEBOOKS"),
    reason="set PYTOF_TEST_NOTEBOOKS=1 to execute the notebooks",
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    # the kernel must import this source tree, not an installed PyToF:
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(
        nb, timeout=600, resources={"metadata": {"path": str(path.parent)}}
    ).execute()
