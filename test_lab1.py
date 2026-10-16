"""Проверки решений на обычных и граничных случаях."""

import unittest

import numpy as np

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput,
    RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, sum_prod, unique_columns, unique_rows,
)
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters,
    is_balanced_number, mse, multiplicative_persistence,
    prime_factorization, pyramid,
)


class PythonBasicsTests(unittest.TestCase):
    def test_count_vowels(self):
        self.assertEqual(count_vowels(TextInput("AEiOuZ")), 5)
        self.assertEqual(count_vowels(TextInput("")), 0)
        self.assertEqual(count_vowels(TextInput("rhythm")), 0)

    def test_unique_characters(self):
        self.assertTrue(has_unique_characters(TextInput("")))
        self.assertTrue(has_unique_characters(TextInput("aA")))
        self.assertFalse(has_unique_characters(TextInput("abba")))

    def test_one_bits(self):
        self.assertEqual(count_one_bits(PositiveIntegerInput(1)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(8)), 1)
        self.assertEqual(count_one_bits(PositiveIntegerInput(255)), 8)

    def test_persistence(self):
        for number, expected in [(4, 0), (39, 3), (999, 4), (10, 1)]:
            self.assertEqual(multiplicative_persistence(PositiveIntegerInput(number)), expected)

    def test_mse(self):
        self.assertEqual(mse(VectorPairInput([1], [3])), 4.0)
        self.assertEqual(mse(VectorPairInput([1, 2], [1, 2])), 0.0)
        self.assertEqual(mse(VectorPairInput([-1, 3], [1, 5])), 4.0)
        with self.assertRaises(ValueError):
            mse(VectorPairInput([], []))

    def test_prime_factorization(self):
        for number, expected in [(1, ""), (2, "(2)"), (13, "(13)"),
                                 (12, "(2**2)(3)"), (86240, "(2**5)(5)(7**2)(11)")]:
            self.assertEqual(prime_factorization(PositiveIntegerInput(number)), expected)

    def test_pyramid(self):
        for number, expected in [(1, 1), (2, "It is impossible"), (5, 2),
                                 (14, 3), (15, "It is impossible")]:
            self.assertEqual(pyramid(PositiveIntegerInput(number)), expected)

    def test_balanced_number(self):
        for number, expected in [(1, True), (12, True), (1234, False),
                                 (1234006, True), (123456, False), (12321, True)]:
            self.assertEqual(is_balanced_number(PositiveIntegerInput(number)), expected)


class NumpyTests(unittest.TestCase):
    def test_sum_prod(self):
        matrices = np.array([np.eye(2), [[2, 0], [0, 3]]])
        vectors = np.array([[[1], [2]], [[3], [4]]])
        np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(matrices, vectors)), [[7], [14]])
        np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(matrices[:1], vectors[:1])), [[1], [2]])

    def test_binarize(self):
        matrix = np.array([[0.4, 0.5, 0.6]])
        np.testing.assert_array_equal(binarize(BinarizeInput(matrix, 0.5)), [[0, 0, 1]])
        np.testing.assert_array_equal(matrix, [[0.4, 0.5, 0.6]])

    def test_unique_rows_columns(self):
        matrix = np.array([[2, 1, 2], [3, 1, 3]])
        self.assertEqual(unique_rows(MatrixInput(matrix)), [[1, 2], [1, 3]])
        self.assertEqual(unique_columns(MatrixInput(matrix)), [[2, 3], [1], [2, 3]])
        self.assertEqual(unique_rows(MatrixInput(np.array([[5]]))), [[5]])

    def test_matrix_statistics(self):
        result = matrix_statistics(RandomMatrixInput(2, 3, seed=7))
        self.assertEqual(result.matrix.shape, (2, 3))
        np.testing.assert_array_equal(result.row_means, result.matrix.mean(axis=1))
        np.testing.assert_array_equal(result.column_means, result.matrix.mean(axis=0))
        np.testing.assert_array_equal(result.row_variances, result.matrix.var(axis=1))
        np.testing.assert_array_equal(result.column_variances, result.matrix.var(axis=0))
        other = matrix_statistics(RandomMatrixInput(2, 3, seed=7))
        np.testing.assert_array_equal(result.matrix, other.matrix)
        single = matrix_statistics(RandomMatrixInput(1, 1, std=0))
        np.testing.assert_array_equal(single.row_variances, [0])

    def test_chess(self):
        np.testing.assert_array_equal(chess(ChessInput(3, 4, 0, 1)),
                                      [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        np.testing.assert_array_equal(chess(ChessInput(1, 1, 5, 9)), [[5]])

    def test_rectangle(self):
        image = draw_rectangle(RectangleInput(2, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertEqual(np.count_nonzero(image[:, :, 0]), 4)
        np.testing.assert_array_equal(image[1:3, 1:3], np.full((2, 2, 3), (255, 0, 0)))
        one = draw_rectangle(RectangleInput(1, 1, 1, 1, (1, 2, 3), (0, 0, 0)))
        np.testing.assert_array_equal(one[0, 0], [1, 2, 3])

    def test_ellipse(self):
        image = draw_ellipse(EllipseInput(2, 1, 5, 5, (255, 0, 0), (0, 0, 0)))
        self.assertEqual(image.shape, (5, 5, 3))
        np.testing.assert_array_equal(image[2, 2], [255, 0, 0])
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(image[2, 0], [255, 0, 0])

    def test_time_series(self):
        result = analyze_time_series(TimeSeriesInput(np.array([1, 3, 1, 0, 2]), 2))
        self.assertEqual(result.mean, 1.4)
        np.testing.assert_array_equal(result.local_maxima_indices, [1])
        np.testing.assert_array_equal(result.local_minima_indices, [3])
        np.testing.assert_array_equal(result.moving_average, [2, 2, 0.5, 1])
        single = analyze_time_series(TimeSeriesInput(np.array([5]), 1))
        np.testing.assert_array_equal(single.local_maxima_indices, [])
        np.testing.assert_array_equal(single.moving_average, [5])

    def test_one_hot(self):
        np.testing.assert_array_equal(one_hot(OneHotInput([0, 2, 3, 0])),
                                      [[1, 0, 0, 0], [0, 0, 1, 0],
                                       [0, 0, 0, 1], [1, 0, 0, 0]])
        self.assertEqual(one_hot(OneHotInput([], 3)).shape, (0, 3))
        np.testing.assert_array_equal(one_hot(OneHotInput([1], 3)), [[0, 1, 0]])
        with self.assertRaises(ValueError):
            one_hot(OneHotInput([3], 3))


if __name__ == "__main__":
    unittest.main()
