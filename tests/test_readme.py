"""The options table in the README is generated, not written by hand."""

from tests.make_readme import README, updated


def test_readme_options_table_is_up_to_date():
    text = README.read_text()
    assert updated(text) == text, (
        "The options table in README.md is out of date; run "
        "`python -m tests.make_readme`."
    )
