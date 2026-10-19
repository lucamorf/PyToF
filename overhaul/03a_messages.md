# 03a — Messaging (replaces all hand-built colour strings)

Commit `f1a9b5c`. Full diff: `03_diff_messages.diff`.

## Before / after
```python
# before: colour codes concatenated by hand, gating by hand (45 prints in 4 files)
if class_obj.opts["verbosity"] > 0:
    print(
        c.INFO
        + "Generated initial conditions for walker number: "
        + c.NUMB
        + str(i)
        + c.ENDC
    )

# after
say(class_obj.opts["verbosity"], 1,
    "Generated initial conditions for walker number: {i}", i=i)
```
Every `{field}` is highlighted as a number automatically. Convergence problems go through `warn(...)`.

## The rules now
| Kind | Mechanism | Examples |
|---|---|---|
| **Warning** (result may be wrong) | `warnings.warn`, always shown, filterable, testable | not converged (`ConvergenceWarning`), radius inconsistent with shape, no error estimates available |
| level 1 notice | `say`, `verbosity>=1` | fallback to trapezoid, R_ref assumed, MCMC progress, boundary conditions of Love numbers |
| level 2 | `verbosity>=2` | timing of relax_to_barotrope/density, rejected MCMC candidates |
| level 3 | `verbosity>=3` | timing of every relax_to_shape |

Colours only on terminals and Jupyter, never in files; `NO_COLOR` / `FORCE_COLOR` honoured. Exception texts no longer contain ANSI codes.

## Behaviour changes (please check each)
1. Convergence/consistency problems are now **warnings** (stderr, `ConvergenceWarning`/`PyToFWarning`) instead of `print` to stdout. Silence with `warnings.simplefilter("ignore", PyToFWarning)`.
2. `relax_to_shape` now warns when it did not converge (previously a transient `\r` message at verbosity>0 only). `relax_to_barotrope/density` use an internal worker, so their deliberate 2-iteration shape steps never warn.
3. `AlgoToF.Algorithm` prints nothing any more; it returns `converged` and `dJ_max` instead (this removed the blank-line printing hack).
4. The Love-number "Boundary condition applied…" message was always printed; now level 1.
5. Messages that said `MaxIterBar` after a *density* relaxation now correctly say `MaxIterDen` (cosmetic bug).
6. `PyToF.color` is gone (was internal).

### `PyToF/messages.py`
```python
"""
Console output of PyToF.

There are two kinds of messages:

* Warnings (:func:`warn`). Something may be wrong with the result, e.g. a
  solver did not converge. They go through the standard :mod:`warnings`
  module, so they are always shown but can be silenced or turned into errors
  with ``warnings.simplefilter``, and ``pytest.warns`` can test for them.

* Progress messages (:func:`say`). Shown only if the ToF option ``verbosity``
  is at least the level of the message:

    1  notices: assumptions made, solver fallbacks, MCMC progress
    2  diagnostics: timings of relax_to_barotrope/density, rejected candidates
    3  timing of every single relax_to_shape call

Messages are written as plain templates; every ``{field}`` is highlighted as a
number, so no colour codes are needed at the call site::

    say(verbosity, 2, "Relaxing to shape took {t:.2e} seconds.", t=elapsed)

The names ``template``, ``verbosity``, ``level``, ``kind``, ``category``,
``stacklevel`` and ``colour`` are parameters of the functions below and
cannot be used as field names.

Colours are used in terminals and Jupyter, not when output is redirected to a
file. Set ``NO_COLOR`` to disable them or ``FORCE_COLOR`` to always use them.
"""

import os
import string
import sys
import warnings

__all__ = ["ConvergenceWarning", "PyToFWarning", "paint", "say", "warn"]

_RESET = "\033[0m"
_NUMBER = "\033[96m"
_KINDS = {
    "info": "\033[94m",
    "warn": "\033[91m",
    "good": "\033[92m",
}


class PyToFWarning(UserWarning):
    """Base class of all warnings issued by PyToF."""


class ConvergenceWarning(PyToFWarning):
    """An iterative procedure did not reach the requested tolerance."""


def _use_colour(stream):
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if "ipykernel" in sys.modules:  # Jupyter renders ANSI colours
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def format_message(template, colour=False, kind="info", **fields):
    """
    Fill ``template`` with ``fields`` (``str.format`` syntax). With
    ``colour=True`` the text gets the colour of ``kind`` and the filled-in
    fields are highlighted as numbers.
    """

    if not colour:
        return template.format(**fields)

    base = _KINDS[kind]
    parts = [base]
    for literal, name, spec, conversion in string.Formatter().parse(template):
        parts.append(literal)
        if name is not None:
            value = string.Formatter().get_field(name, (), fields)[0]
            if conversion:
                value = string.Formatter().convert_field(value, conversion)
            parts.append(_NUMBER + format(value, spec or "") + base)
    parts.append(_RESET)

    return "".join(parts)


def paint(text, kind="info"):
    """Return ``text`` in the colour of ``kind`` if colours are in use."""

    if not _use_colour(sys.stdout):
        return text

    return _KINDS[kind] + text + _RESET


def say(verbosity, level, template, kind="info", **fields):
    """
    Print ``template`` if ``verbosity >= level``. ``kind`` is "info", "warn"
    or "good" and only chooses the colour; use :func:`warn` for problems that
    must not be missed.
    """

    if verbosity < level:
        return

    print(
        format_message(
            template,
            colour=_use_colour(sys.stdout),
            kind=kind,
            **fields,
        )
    )


def warn(template, category=PyToFWarning, stacklevel=2, **fields):
    """
    Issue a warning with the text ``template`` (``str.format`` syntax).
    ``stacklevel=2`` attributes the warning to the caller of the function
    calling this one, which is usually the user's code.
    """

    warnings.warn(
        format_message(template, **fields),
        category,
        stacklevel=stacklevel + 1,
    )
```
