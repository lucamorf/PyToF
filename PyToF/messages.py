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

__all__ = [
    "ConvergenceWarning",
    "PyToFWarning",
    "RenamedWarning",
    "paint",
    "say",
    "warn",
]

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


class RenamedWarning(PyToFWarning, FutureWarning):
    """
    A name was changed. Being a FutureWarning it is shown by default, also
    outside of ``__main__`` (DeprecationWarnings are hidden for most code).
    """


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
