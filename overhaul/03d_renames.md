# 03d — Naming: DONE (all approved by you), old names still work

Every renamed public name keeps working and emits a `RenamedWarning` (a `FutureWarning`, shown by default) that points at the user's line.

## Done
| Was | Now | Commit |
|---|---|---|
| `MaxIterShape` / `MaxIterUpdate` / `MaxIterBar` / `MaxIterDen` | `max_iter_shape` / `max_iter_update` / `max_iter_barotrope` / `max_iter_density` | `d5a8551` |
| `Target_Js` / `Sigma_Js` | `target_Js` / `sigma_Js` | `d5a8551` |
| `rho_MAX` / `P_MAX` | `rho_max` / `P_max` | `d5a8551` |
| `X._set_IC()` | `X.reset()` | `2cd9e8f` |
| `X.Js_error` | `X.Js_errors` | `2cd9e8f` |
| `bugfix_iter` (internal) | `_debug_iteration` | `2cd9e8f` |
| first parameter `class_obj` | `self` in ToF methods, `tof` in module-level helpers | `8878964` |
| `X.ss` / `X.SS` | `X.figure_functions` / `X.volume_integrals` | latest |

## Deliberately unchanged
- `Period` (not renamed, by agreement), `N, G, H, P0, R_ref, R_phys, M_phys`, `li, rhoi, Pi, Js, A0, U, R_eq_to_R_m, ...` (paper notation).
- `ss`/`SS`/`s2`/`S0`... **inside `AlgoToF.py`** and the generated polynomials: they match the Mathematica notebook and the literature term by term.
- Module file names (`ClassToF.py` etc.): `from PyToF import ToF` is the documented entry point; the notebooks' `from PyToF import ClassToF` keeps working.

## Notes
- `volume_integrals` holds 2*(order+1) arrays: S_n followed by the primed S'_n.
- Notebooks still use some old names (`MaxIterBar`, `Target_Js`, `Sigma_Js`, `_set_IC`, `.ss`); they run but print `RenamedWarning`. They get updated in steps 08-10.
