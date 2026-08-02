import pytest

ROSENBROCK = "100*(x2 - x1^2)^2 + (1 - x1)^2"


def test_fmindfp_imports():
    from dfp_formula_project import fmindfp

    assert callable(fmindfp.fmindfp)


def test_fmindfp_minimizes_rosenbrock():
    from dfp_formula_project.fmindfp import fmindfp
    from dfp_formula_project.function import Function

    fun = Function(ROSENBROCK)

    # Default arguments return the 3-tuple (x, allvecs, lst). main.py and gui.py
    # both index this, so the shape is asserted here to pin the interface.
    result = fmindfp(fun, [0.4, -0.6], maxiter=10000, disp=False)

    assert isinstance(result, tuple)
    assert len(result) == 3

    x, allvecs, log = result

    assert x == pytest.approx([1.0, 1.0], abs=1e-3)
    assert fun(x) == pytest.approx(0.0, abs=1e-6)
    assert len(allvecs) > 1
    assert allvecs[-1] == pytest.approx(x)
