"""Unit tests for Step 4 helpers."""
import numpy as np


def test_border_elems_generic(step4):
    a = np.arange(25).reshape(5, 5)
    border = step4.border_elems_generic(a, 1)
    assert sorted(border) == sorted(
        set(a[0]) | set(a[-1]) | set(a[:, 0]) | set(a[:, -1])
    )
    assert len(step4.border_elems_generic(a, 2)) == 24  # all but the centre
