"""Work with (p, 0) SU(3) irreps in the most inefficient way possible."""

import abc
from itertools import chain, combinations, permutations
from functools import reduce
from math import factorial, prod
from typing import Iterable, List, Tuple

import attr
import numpy as np


class SL3CRep(abc.ABC):
    """Special-linear Lie algebra of 3 complex dimensions."""
    @abc.abstractmethod
    def get_diagonal_op(
        self,
        positive_index: int,
        negative_index: int
    ) -> np.ndarray:
        """Make representative of operator diagonal in standard representation.

        Parameters
        ----------
        positive_index
            Index for the basis element with a +1 on the diagonal in the
            standard representation.
        negative_index
            Index for the basis element with a -1 on the diagonal in the
            standard representation.
        Returns
        -------
            Representative of the specified Lie-algebra element.
        """

    @abc.abstractmethod
    def get_lowering_op(self, low_index: int, high_index: int) -> np.ndarray:
        """Make representative of a lowering operator in standard representation.

        Parameters
        ----------
        low_index
            The basis element to lower to.
        high_index
            The basis element to lower from.
        Returns
        -------
            Representative of the specified Lie-algebra element.
        """

    @abc.abstractmethod
    def get_raising_op(self, low_index: int, high_index: int) -> np.ndarray:
        """Make representative of a raising operator in standard representation.

        Parameters
        ----------
        low_index
            The basis element to lower to.
        high_index
            The basis element to lower from.
        Returns
        -------
            Representative of the specified Lie-algebra element.
        """

    def get_basis(self) -> List[np.ndarray]:
        """Return all the basis elements."""
        diagonal_ops = [
            self.get_diagonal_op(
                positive_index=positive_index,
                negative_index=positive_index + 1,
            )
            for positive_index in (0, 1)
        ]
        lowering_ops = [
            self.get_lowering_op(
                    low_index=low_index,
                    high_index=high_index,
                )
            for low_index, high_index in combinations((0, 1, 2), 2)
        ]
        raising_ops = [
            self.get_raising_op(
                    low_index=low_index,
                    high_index=high_index,
                )
            for low_index, high_index in combinations((0, 1, 2), 2)
        ]
        return diagonal_ops + lowering_ops + raising_ops

    def __repr__(self) -> str:
        return '\n\n'.join(map(str, self.get_basis()))


@attr.s(auto_attribs=True, repr=False)
class StandardSL3CRep(SL3CRep):
    """Standard three-dimensional representation."""
    basis: np.ndarray = attr.ib(factory=lambda: np.eye(3, dtype=int))

    def get_diagonal_op(
        self,
        positive_index: int,
        negative_index: int
    ) -> np.ndarray:
        return (
            np.outer(self.basis[positive_index], self.basis[positive_index]) -
            np.outer(self.basis[negative_index], self.basis[negative_index])
        )

    def get_lowering_op(self, low_index: int, high_index: int) -> np.ndarray:
        return np.outer(self.basis[low_index], self.basis[high_index])

    def get_raising_op(self, low_index: int, high_index: int) -> np.ndarray:
        return np.outer(self.basis[high_index], self.basis[low_index])


def get_product_rule_lie_algebra_representative(
    representatives: List[np.ndarray]
) -> np.ndarray:
    """Get the tensor product representative of a collection of representatives.

    Parameters
    ----------
    representatives
        All the component representatives of the Lie-algebra element.

    Returns
    -------
        The representative AII...I + IBI...I + ... of the collection of
        representatives [A, B, ...] in the tensor-product representation.
    """
    assert all(
        representative.shape[0] == representative.shape[1] and
        len(representative.shape) == 2
        for representative in representatives
    )
    identities = [np.eye(representative.shape[0], dtype=representative.dtype)
                  for representative in representatives]
    return sum(
        reduce(
            np.kron,
            [identity if idx != term_idx else representative
             for idx, identity in enumerate(identities)]
        )
        for term_idx, representative in enumerate(representatives)
    )


@attr.s(auto_attribs=True, repr=False)
class DualSL3CRep(SL3CRep):
    """Dual of an SL3C representation."""
    original_sl3c_rep: SL3CRep

    def get_diagonal_op(
        self,
        positive_index: int,
        negative_index: int
    ) -> np.ndarray:
        return -self.original_sl3c_rep.get_diagonal_op(
            positive_index=positive_index,
            negative_index=negative_index,
        ).T

    def get_lowering_op(
        self,
        low_index: int,
        high_index: int
    ) -> np.ndarray:
        return -self.original_sl3c_rep.get_lowering_op(
            low_index=low_index,
            high_index=high_index,
        ).T

    def get_raising_op(
        self,
        low_index: int,
        high_index: int,
    ) -> np.ndarray:
        return -self.original_sl3c_rep.get_raising_op(
            low_index=low_index,
            high_index=high_index
        ).T


@attr.s(auto_attribs=True, repr=False)
class TensorProductSL3CRep(SL3CRep):
    """Tensor product of several representations."""
    generating_reps: List[SL3CRep]

    def get_diagonal_op(
        self,
        positive_index: int,
        negative_index: int,
    ) -> np.ndarray:
        representatives = [rep.get_diagonal_op(positive_index, negative_index)
                           for rep in self.generating_reps]
        return get_product_rule_lie_algebra_representative(representatives)

    def get_lowering_op(self, low_index: int, high_index: int) -> np.ndarray:
        representatives = [
            rep.get_lowering_op(
                low_index=low_index,
                high_index=high_index,
            )
            for rep in self.generating_reps
        ]
        return get_product_rule_lie_algebra_representative(representatives)

    def get_raising_op(self, low_index: int, high_index: int) -> np.ndarray:
        representatives = [
            rep.get_raising_op(
                low_index=low_index,
                high_index=high_index,
            )
            for rep in self.generating_reps
        ]
        return get_product_rule_lie_algebra_representative(representatives)


def get_iterative_raising_operator_applications(
        sl3c_rep: SL3CRep,
        initial_vector: np.ndarray,
) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
    """Repeat raising operators on an initial vector until annihilation."""
    raising_operators = {
        raising_pair: sl3c_rep.get_raising_op(*raising_pair)
        for raising_pair in ((0, 1), (1, 2))
    }
    image_vectors = []
    applied_raising_sequences = []
    most_recently_generated_vectors = [initial_vector]
    most_recently_applied_raising_sequences = [[]]
    while most_recently_generated_vectors:
        actively_generated_vectors = []
        actively_applied_raising_sequences = []
        for raising_pair, raising_operator in raising_operators.items():
            for vector, applied_sequence in zip(
                    most_recently_generated_vectors,
                    most_recently_applied_raising_sequences,
            ):
                image_vector = raising_operator @ vector
                if np.linalg.norm(image_vector) != 0:
                    actively_generated_vectors.append(image_vector)
                    actively_applied_raising_sequences.append(
                        applied_sequence + [raising_pair]
                    )
        image_vectors += most_recently_generated_vectors
        applied_raising_sequences += most_recently_applied_raising_sequences
        most_recently_generated_vectors = actively_generated_vectors
        most_recently_applied_raising_sequences =\
            actively_applied_raising_sequences
    return image_vectors, applied_raising_sequences


def make_projector_out_of_image_vectors(
        image_vectors: List[np.ndarray],
        zero_threshold: float = 1e-7,
) -> np.ndarray:
    """Make projector onto the linear span of given vectors.

    Parameters
    ----------
    image_vectors
        The vectors spanning the space to be projected onto.
    zero_threshold
        Threshold to use when determining linear dependence.

    Returns
    -------
        The orthogonal projector onto the linear span.
    """
    left_singular_column_vectors, singular_values, _ = np.linalg.svd(
        np.array(image_vectors).T
    )
    return np.array([left_singular_column_vectors[:, idx]
                     for idx in range(singular_values.size)
                     if singular_values[idx] > zero_threshold])


def kron(factors: Iterable[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, factors, np.array(1))


def symmetrize_trit_vector_slow_unnorm(
    multiplicities: Tuple[int, int, int],
) -> np.ndarray:
    kets = np.eye(3, dtype=float)
    canonical_ordering = sum(
        [multiplicity * [idx] for idx, multiplicity
         in enumerate(multiplicities)],
        [],
    )
    return sum(
        kron(kets[basis_idx] for basis_idx in ordering)
        for ordering in permutations(canonical_ordering)
    )


def symmetrization_norm_squared(multiplicities: Tuple[int, int, int]) -> int:
    return (
        factorial(sum(multiplicities)) * prod(map(factorial, multiplicities))
    )


def get_symmetric_su3_matrix_element_slow(
    left_multiplicities: Tuple[int, int, int],
    unitary_matrix: np.ndarray,
    right_multiplicities: Tuple[int, int, int],
) -> complex:
    left_tensor_power = sum(left_multiplicities)
    right_tensor_power = sum(right_multiplicities)
    if left_tensor_power != right_tensor_power:
        raise ValueError('Left and right tensor powers should be equal.')
    tensor_power = left_tensor_power
    left_symmetrized_vector = symmetrize_trit_vector_slow_unnorm(
        left_multiplicities
    )
    right_symmetrized_vector = symmetrize_trit_vector_slow_unnorm(
        right_multiplicities
    )
    left_norm_squared = symmetrization_norm_squared(left_multiplicities)
    right_norm_squared = symmetrization_norm_squared(right_multiplicities)
    return (
        left_symmetrized_vector @ kron(tensor_power * [unitary_matrix])
        @ right_symmetrized_vector / np.sqrt(
            left_norm_squared * right_norm_squared
        )
    )


def partitions(total: int, boxes: int) -> Iterable[Tuple[int, ...]]:
    if boxes == 1:
        return [(total,)]
    return chain(
        *[[partition + (last_box_pop,) for partition
           in partitions(total - last_box_pop, boxes - 1)]
          for last_box_pop in range(total + 1)]
    )


def make_su3_unitary_representative_slow(
    unitary_matrix: np.ndarray,
    symmetrized_copies: int,
) -> np.ndarray:
    return np.array(
        [[get_symmetric_su3_matrix_element_slow(
            left_multiplicities,
            unitary_matrix,
            right_multiplicities,
        )
          for right_multiplicities in partitions(symmetrized_copies, 3)]
         for left_multiplicities in partitions(symmetrized_copies, 3)]
    )
