"""
Reference solutions used by the accuracy notebooks and the test suite.

Every number in this file comes from the literature; nothing is fitted to
PyToF. The tests (``tests/cases.py``) import ``Bessel`` from here, so the
notebooks and the tests cannot disagree.
"""

import numpy as np
import scipy.special

JUPITER_RADIUS = (
    71492000.0  # m, equatorial radius used for the J_2n of Jupiter
)

# --------------------------------------------------------------------------
# Gravitational moments of Jupiter-like polytropes
# --------------------------------------------------------------------------
# Wisdom, J. and Hubbard, W. B., "Differential rotation in Jupiter: A
# comparison of methods", Icarus 267, 315-322 (2016),
# doi:10.1016/j.icarus.2015.12.030.
#
# Rotating n=1 polytrope (P = rho^2), solid-body rotation, m = 0.089195487
# (the Jupiter value used in the paper), closed-form (Bessel function)
# solution. Entries: [R_eq/R_m, J2, J4, ..., J20].
Bessel = np.array(
    [
        1.022875431133174e0,
        1.398851089834702e-2,
        -5.318281001092907e-4,
        3.011832290533641e-5,
        -2.132115710725050e-6,
        1.740671195871128e-7,
        -1.568219505602588e-8,
        1.518099230068580e-9,
        -1.551985393081485e-10,
        1.655948019619652e-11,
        -1.829544870258362e-12,
    ]
)

# Same paper: the Concentric Maclaurin Spheroid method (CMS) with barotropic
# differential rotation, two different wind profiles (DR2: A2..A8 below,
# DR3: A2..A20 below). R_eq/R_m is not given (nan).
CMSDR2 = np.array(
    [
        np.nan,
        1.399685529663848e-2,
        -5.358080750264726e-4,
        3.159100841281628e-5,
        -2.626580737287536e-6,
        2.745394255627955e-7,
        -3.247718410175606e-8,
        4.106439257702602e-9,
        -5.430942499498113e-10,
        7.456525895586537e-11,
        -1.057232912152619e-11,
    ]
)
CMSDR3 = np.array(
    [
        np.nan,
        1.399739489765669e-2,
        -5.358967290243679e-4,
        3.163072519976169e-5,
        -2.715682955893595e-6,
        3.482051102658201e-7,
        1.908012504014946e-8,
        -1.019589599416940e-7,
        1.028205395308815e-7,
        -2.636937795839493e-8,
        -3.593890869111017e-8,
    ]
)

# Differential rotation parameters of the two wind profiles above, in
# m^2 s^-2 (the `alphas` option of PyToF), taken from the same paper.
ALPHAS_DR2 = (
    np.array(
        [
            -5.8954166431924311100e-2,
            4.6123389620922838894e-1,
            -1.0426642770099761037e0,
            8.3030636453130002295e-1,
        ]
    )
    * 1e6
)
ALPHAS_DR3 = (
    np.array(
        [
            -5.4553556626624011283e-1,
            2.0659730767589977063e1,
            -2.9751201811994826585e2,
            2.1388899216795743996e3,
            -8.7087211563139935606e3,
            2.1342828803262666042e4,
            -3.2058440847282883624e4,
            2.8850390979208579665e4,
            -1.4264411879490267893e4,
            2.9770389071934714593e3,
        ]
    )
    * 1e6
)


def alphas_array(alphas):
    """Pad a list of alphas with zeros to the length PyToF expects (12)."""
    out = np.zeros(12)
    out[: len(alphas)] = alphas
    return out


# --------------------------------------------------------------------------
# Normalized moment of inertia, I / (M R^2), of simple bodies
# --------------------------------------------------------------------------
# Spherical case, textbook values. For a rotating body of constant density the
# moment of inertia scales with the squared equatorial radius, which is why
# the notebooks multiply with R_eq_to_R_m**2.
NMOI_SPHERE_CONSTANT_DENSITY = 2 / 5
NMOI_SPHERE_LINEAR_DENSITY = 4 / 15  # rho = rho_c * r / R


# --------------------------------------------------------------------------
# Love functions K_n of spheres with analytical density profiles
# --------------------------------------------------------------------------
# Gavrilov, S. V., Zharkov, V. N. and Leontev, V. V., "Influence of tides on
# the gravitational field of Jupiter", Soviet Astronomy 19, 618-621 (1976).
# Three density profiles with closed-form (or rapidly converging series)
# solutions of the perturbation equation:
#   "linear":     rho = 4 rho_mean (1 - x)
#   "quadratic":  rho = 2.5 rho_mean (1 - x^2)
#   "polytropic": rho = pi^2/3 rho_mean sin(pi x) / (pi x)       (n = 1)
# with x = r/R.


def love_cases(rho_mean, radius, G, n_terms=50):
    """
    Analytical density, gravity, T_n and K_n of the three profiles.

    Returns a dict name -> dict(rho=f(r), g=f(r), T=f(r, n), K=f(r, n)),
    all in SI units, r an array in m, n the degree (>= 2) of the Love
    function.
    """

    def _series(n, ratio, step, num_k, den_k):
        # P[i] = P[i-1] * Delta(k = i-1), the recurrence of the power series
        # solutions of the linear and quadratic profiles.
        P = np.ones(n_terms)
        for i in range(1, n_terms):
            P[i] = P[i - 1] * num_k(i - 1, n) / den_k(i - 1, n)
        i = np.arange(1, n_terms)
        denom = (2 * n + 1) + np.sum(ratio**i * (step * i + 2 * n + 1) * P[1:])
        return P, denom

    lin = (
        lambda n: _series(
            n,
            3 / 4,
            1,
            lambda k, n: k**2 + (2 * n + 1) * k - 4,
            lambda k, n: k**2 + (2 * n + 3) * k + 2 * (n + 1),
        ),
        3 / 4,
        1,
    )
    quad = (
        lambda n: _series(
            n,
            3 / 5,
            2,
            lambda k, n: 2 * k**2 + (2 * n + 1) * k - 5,
            lambda k, n: 2 * k**2 + (2 * n + 5) * k + 2 * n + 3,
        ),
        3 / 5,
        2,
    )

    def series_case(spec, g_func):
        build, ratio, step = spec

        def T(r, n):
            x = r / radius
            P, denom = build(n)
            C0 = (2 * n + 1) * g_func(radius) * radius / denom
            Ci = C0 * ratio ** np.arange(n_terms) * P
            return sum(Ci[i] * x ** (n + step * i) for i in range(n_terms))

        def K(r, n):
            x = r / radius
            P, denom = build(n)
            s = sum(
                ratio**i * P[i] * x ** (step * i) for i in range(1, n_terms)
            )
            return x**n * ((2 * n + 1) * (1.0 + s) / denom - 1.0)

        return T, K

    def g_linear(r):
        x = r / radius
        return 16 / 3 * np.pi * G * rho_mean * radius * x * (1 - 3 * x / 4)

    def g_quadratic(r):
        x = r / radius
        return 10 * np.pi * G * rho_mean * radius * x * (1 / 3 - x**2 / 5)

    def g_polytropic(r):
        x = r / radius

        def dsinc(y):  # derivative of sinc
            return (y * np.cos(y) - np.sin(y)) / y**2

        return (
            -4 / 3 * np.pi * G * radius**3 * rho_mean
            * dsinc(np.pi * x) * np.pi / radius**2
        )  # fmt: skip

    jv = scipy.special.jv

    def T_polytropic(r, n):
        x = r / radius
        return (
            (2 * n + 1) * radius * g_polytropic(radius)
            / (np.pi * jv(n - 0.5, np.pi))
            * x**-0.5 * jv(n + 0.5, np.pi * x)
        )  # fmt: skip

    def K_polytropic(r, n):
        x = r / radius
        return (2 * n + 1) * jv(n + 0.5, np.pi * x) / (
            np.pi * x**0.5 * jv(n - 0.5, np.pi)
        ) - x**n

    T_lin, K_lin = series_case(lin, g_linear)
    T_quad, K_quad = series_case(quad, g_quadratic)
    return {
        "linear": dict(
            rho=lambda r: 4 * rho_mean * (1 - r / radius),
            g=g_linear, T=T_lin, K=K_lin,
        ),
        "quadratic": dict(
            rho=lambda r: 2.5 * rho_mean * (1 - (r / radius) ** 2),
            g=g_quadratic, T=T_quad, K=K_quad,
        ),
        "polytropic": dict(
            rho=lambda r: np.pi**2 / 3 * rho_mean
            * np.sin(np.pi * r / radius) / (np.pi * r / radius),
            g=g_polytropic, T=T_polytropic, K=K_polytropic,
        ),
    }  # fmt: skip
