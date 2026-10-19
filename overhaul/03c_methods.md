# 03c — Methods instead of `functools.partial`

Commit `1c04672`. Full diff: `03_diff_methods.diff`.

**Before:** every `ToF()` ran 16 local imports and 16 `self.foo = functools.partial(foo, self)`; no autocomplete or static analysis, `plot_autocorr` bound differently, `help(X.relax_to_shape)` showed a `partial`.

**After:** the functions stay where they are and are attached once, at class level. They take the ToF object as their first argument, so they simply *are* methods. Measured: no per-instance methods any more (`[k for k in vars(X) if callable(...)] == []`), instance pickles smaller, parallel MCMC test passes.

```python
class ToF:
    # Shape and density relaxation:
    relax_to_shape = relax_to_shape
    relax_to_barotrope = relax_to_barotrope
    ...
    plot_autocorr = staticmethod(plot_autocorr)  # does not take a ToF object
```
(imports now at the top of `ClassToF.py`; there is no import cycle.)

## Not done yet (needs your OK, see 03d)
The functions still call their first parameter `class_obj` (630 occurrences). Renaming it to `self` and moving them physically into the class body/mixins is purely mechanical but touches almost every line, so I did not do it without asking.
