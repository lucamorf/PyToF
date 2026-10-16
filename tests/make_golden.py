"""
Regenerate tests/golden/golden.json from the *current* code.

Run deliberately (`python -m tests.make_golden`), only when a physics change is
intended, and review the resulting git diff of golden.json before committing.
"""

import json
from pathlib import Path

from tests.cases import GOLDEN_CASES, summarize

OUT = Path(__file__).parent / "golden" / "golden.json"


def main():
    golden = {name: summarize(build()) for name, build in GOLDEN_CASES.items()}
    OUT.write_text(json.dumps(golden, indent=1, sort_keys=True))
    print(f"wrote {len(golden)} cases to {OUT}")


if __name__ == "__main__":
    main()
