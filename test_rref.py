import pytest
import numpy as np

from random import randint
from sympy import Matrix

from rref import rref


@pytest.fixture
def augmented_and_rref() -> list[tuple[list, list]]:
    """Return a list of tuples of augmented matrices and their rrefs."""
    return [
        (
            [[1, 2, 4, 5], [2, 4, 5, 4], [4, 5, 4, 2]],
            [[1, 0, 0, 1], [0, 1, 0, -2], [0, 0, 1, 2]],
        ),
        ([[1, 2, 7], [-2, 5, 4], [-5, 6, -3]], [[1, 0, 3], [0, 1, 2], [0, 0, 0]]),
        (
            [[1, 0, 5, 2], [-2, 1, -6, -1], [0, 2, 8, 6]],
            [[1, 0, 5, 2], [0, 1, 4, 3], [0, 0, 0, 0]],
        ),
        (
            [[1, 0, -6, 21], [4, 2, -11, 55], [0, 1, 3, -4]],
            [[1, 0, 0, 3], [0, 1, 0, 5], [0, 0, 1, -3]],
        ),
        (
            [[3, -2, 8, 0], [9, -6, 24, 0], [6, -4, 16, 0]],
            [[1, -(2 / 3), (8 / 3), 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ),
    ]


def test_hand(augmented_and_rref: list[tuple[list, list]]) -> None:
    """Test a bunch of hand-calculated rrefs."""
    for augmented_matrix, true_rref in augmented_and_rref:
        augmented_matrix = np.array(augmented_matrix, dtype=np.float64)
        assert np.isclose(rref(augmented_matrix), true_rref).all()


def test_sympy() -> None:
    """Test that our rref agrees with sympy for random matrices."""
    for i in range(1_000):
        nrows = randint(2, 12)
        ncols = nrows + randint(1, 5)

        augmented_matrix = np.random.randint(-100, 100, size=(nrows, ncols))
        augmented_matrix = augmented_matrix.astype(np.float64)

        sympy_rref = Matrix(augmented_matrix).rref(pivots=False)
        our_rref = rref(augmented_matrix)

        assert np.isclose(sympy_rref, our_rref).all()
