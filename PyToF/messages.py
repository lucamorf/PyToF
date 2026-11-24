"""
Console output of PyToF.

There are two kinds of messages:

* Warnings (:func:`warn`). Something may be wrong with the result, e.g. a
  solver did not converge. They go through the standard :mod:`warnings`
  module, so they are always shown but can be silenced or turned into errors
  with ``warnings.simplefilter``, and ``pytest.warns`` can test for them.

  Anything that can change the result belongs here, e.g. the fallback to the
  trapezoid rule (:class:`AccuracyWarning`), never in a progress message.

* Progress messages (:func:`say`). Shown only if the ToF option ``verbosity``
  is at least the level of the message:

    1  summary: one line per relax_to_shape/barotrope/density call with the
       number of iterations and the final convergence measures
    2  details: timings, atmosphere transition index, boundary conditions of
       the Love number calculation
    3  trace: one line per iteration of the relaxation loops and every
       renormalisation of radius and mass

Messages are written as plain templates; the text keeps the terminal's colour
and every numeric ``{field}`` is highlighted, integers (counts such as
iterations) in a different colour than floats (results), so no colour codes
are needed at the call site::

    say(verbosity, 2, "Relaxing to shape took {t:.2e} seconds.", t=elapsed)

The names ``template``, ``verbosity``, ``level``, ``kind``, ``category``,
``stacklevel`` and ``colour`` are parameters of the functions below and
cannot be used as field names.

Colours are used in terminals and Jupyter, not when output is redirected to a
file. Set ``NO_COLOR`` to disable them or ``FORCE_COLOR`` to always use them.
"""

import numbers
import os
import string
import sys
import warnings

__all__ = [
    "AccuracyWarning",
    "ConvergenceWarning",
    "PressureError",
    "PyToFError",
    "PyToFWarning",
    "RenormalisationError",
    "RenamedWarning",
    "paint",
    "say",
    "warn",
    "warn_simpson_fallback",
    "warn_surface_slope",
]

_RESET = "\033[0m"
_NUMBER = "\033[96m"  # floats: results
_COUNT = "\033[93m"  # integers: counts such as iterations
_KINDS = {
    "info": "",  # plain text, only the numbers are highlighted
    "warn": "\033[91m",
    "good": "\033[92m",
}


class PyToFError(RuntimeError):
    """Base class of the errors raised by PyToF at run time."""


class PressureError(PyToFError):
    """
    The hydrostatic pressure profile is unphysical (negative or not
    increasing inwards), e.g. because Simpson's rule failed with
    ``simpson_check=False``.
    """


class RenormalisationError(PyToFError):
    """
    Rescaling the level surfaces or the densities did not reproduce the
    physical radius or mass that was asked for.
    """


class PyToFWarning(UserWarning):
    """Base class of all warnings issued by PyToF."""


class ConvergenceWarning(PyToFWarning):
    """An iterative procedure did not reach the requested tolerance."""


class AccuracyWarning(PyToFWarning):
    """The result is less accurate than requested, e.g. after a fallback."""


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
    ``colour=True`` the text gets the colour of ``kind`` ("info" is the
    terminal's own colour) and numeric fields are highlighted: integers
    (counts) and other numbers (results) in different colours.
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
            text = format(value, spec or "")
            if isinstance(value, numbers.Integral) and not isinstance(
                value, bool
            ):
                text = _COUNT + text + (base or _RESET)
            elif isinstance(value, numbers.Real):
                text = _NUMBER + text + (base or _RESET)
            parts.append(text)
    parts.append(_RESET)

    return "".join(parts)


def paint(text, kind="info"):
    """Return ``text`` in the colour of ``kind`` if colours are in use."""

    if not _use_colour(sys.stdout) or not _KINDS[kind]:
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


def warn_simpson_fallback(reason):
    """
    Warn that PyToF gave up Simpson's rule for the trapezoid rule. ``reason``
    completes the sentence "... because <reason>".
    """

    warn(
        "Reverting to the more stable and less accurate trapezoid rule for "
        "integration because {reason}",
        AccuracyWarning,
        stacklevel=2,
        reason=reason,
    )


def warn_surface_slope(exponent, threshold):
    """
    Warn that the density has a (nearly) infinite slope at the surface, which
    Simpson's rule handles badly. ``exponent`` is the estimated n of
    rho ~ (R - r)^n, ``threshold`` the option surface_exponent_warn.
    """

    warn(
        "The density near the surface behaves like (R - r)^{exponent:.2f}, "
        "below the limit {threshold:.2f} of the option surface_exponent_warn: "
        "its slope is (nearly) infinite there. Simpson's rule may then "
        "converge slowly or not at all (ConvergenceWarning); try "
        "use_simpson=False.",
        AccuracyWarning,
        stacklevel=2,
        exponent=exponent,
        threshold=threshold,
    )
