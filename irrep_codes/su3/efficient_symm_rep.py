"""Work with (p, 0) SU(3) irreps efficiently."""

from itertools import product
from math import prod
from typing import List, Tuple

import numpy as np
from scipy.special import factorial

from irrep_codes.su3.combinatorics import partitions


def generate_row_col_sum_constrained_posint_matrices(
    row_sums: Tuple[int, int, int],
    col_sums: Tuple[int, int, int],
) -> List[np.ndarray]:
    row_sorted_idxs = np.argsort(row_sums)
    col_sorted_idxs = np.argsort(col_sums)
    sorted_row_sums = np.array(row_sums)[row_sorted_idxs]
    sorted_col_sums = np.array(col_sums)[col_sorted_idxs]
    max_upper_left = min(sorted_row_sums[0], sorted_col_sums[0])
    matrices = []
    for upper_left in range(max_upper_left + 1):
        for upper_middle, middle_left in product(
            range(min(sorted_col_sums[1] + 1, sorted_row_sums[0] - upper_left + 1)),
            range(min(sorted_row_sums[1] + 1, sorted_col_sums[0] - upper_left + 1)),
        ):
            if (
                sorted_col_sums[2] >= sorted_row_sums[0] - upper_left - upper_middle
            ) and (sorted_row_sums[2] >= sorted_col_sums[0] - upper_left - middle_left):
                for middle_middle in range(
                    min(
                        sorted_col_sums[1] - upper_middle,
                        sorted_row_sums[1] - middle_left,
                    )
                    + 1
                ):
                    sub_matrix = np.array(
                        [[upper_left, upper_middle], [middle_left, middle_middle]]
                    )
                    sub_col_sums = sub_matrix.sum(axis=0)
                    sub_row_sums = sub_matrix.sum(axis=1)
                    sub_lower_row = sorted_col_sums[:-1] - sub_col_sums
                    sub_right_col = sorted_row_sums[:-1] - sub_row_sums
                    right_col_deficiency = sorted_col_sums[-1] - sub_right_col.sum()
                    lower_row_deficiency = sorted_row_sums[-1] - sub_lower_row.sum()
                    if 0 <= right_col_deficiency == lower_row_deficiency:
                        sorted_matrix = np.vstack(
                            [
                                np.hstack([sub_matrix, sub_right_col[:, np.newaxis]]),
                                np.hstack([sub_lower_row, lower_row_deficiency]),
                            ]
                        )
                        matrices.append(
                            sorted_matrix[
                                np.ix_(
                                    np.argsort(row_sorted_idxs),
                                    np.argsort(col_sorted_idxs),
                                )
                            ]
                        )
    return matrices


def get_overlap_term(unitary_matrix: np.ndarray, overlap_matrix: np.ndarray) -> complex:
    return np.prod(unitary_matrix**overlap_matrix / factorial(overlap_matrix))


def get_symmetric_su3_matrix_element(
    left_multiplicities: Tuple[int, int, int],
    unitary_matrix: np.ndarray,
    right_multiplicities: Tuple[int, int, int],
) -> complex:
    prefactor = prod(
        np.sqrt(factorial(multiplicity))
        for multiplicity in left_multiplicities + right_multiplicities
    )
    return prefactor * sum(
        get_overlap_term(unitary_matrix, overlap_matrix)
        for overlap_matrix in generate_row_col_sum_constrained_posint_matrices(
            row_sums=left_multiplicities, col_sums=right_multiplicities
        )
    )


def make_su3_unitary_representative(
    unitary_matrix: np.ndarray, symmetrized_copies: int
) -> np.ndarray:
    return np.array(
        [
            [
                get_symmetric_su3_matrix_element(
                    left_multiplicities, unitary_matrix, right_multiplicities
                )
                for right_multiplicities in partitions(symmetrized_copies, 3)
            ]
            for left_multiplicities in partitions(symmetrized_copies, 3)
        ]
    )
