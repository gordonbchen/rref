import numpy as np


def get_leading_zeros(array: np.ndarray) -> np.ndarray:
    """Return an array of the number of leading zeros in each row."""
    leading_zeros = np.zeros(array.shape[0], dtype=np.int32)
    for nrow, row in enumerate(array):
        for ncol, num in enumerate(row):
            if num != 0:
                break
        leading_zeros[nrow] = ncol

    return leading_zeros


def rref(augmented_matrix: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Return the reduced row echelon form of the augmented matrix."""
    for nrow in range(augmented_matrix.shape[0]):
        if verbose:
            print("Start")
            print(augmented_matrix)

        # Transpose leading zeros.
        leading_zeros = get_leading_zeros(augmented_matrix)
        inds = leading_zeros.argsort()
        augmented_matrix = augmented_matrix[inds]

        if verbose:
            print("Transposed")
            print(augmented_matrix)

        # Convert leading number to leading 1.
        if augmented_matrix[nrow, nrow] != 0:
            augmented_matrix[nrow] /= augmented_matrix[nrow, nrow]

            if verbose:
                print("Leading 1")
                print(augmented_matrix)

        # Make other rows have 0 in that column.
        other_rows = [i for i in range(augmented_matrix.shape[0]) if i != nrow]
        augmented_matrix[other_rows] -= (
            augmented_matrix[nrow] * augmented_matrix[other_rows, nrow : nrow + 1]
        )

        if verbose:
            print("Column 0s")
            print(augmented_matrix, "\n")

    return augmented_matrix


if __name__ == "__main__":
    augmented_matrix = np.array(
        [[1, 2, 4, 5], [2, 4, 5, 4], [4, 5, 4, 2]], dtype=np.float64
    )
    print(rref(augmented_matrix, verbose=True))
