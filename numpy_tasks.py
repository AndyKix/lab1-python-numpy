"""Решения задач по NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum(np.matmul(matrices, vectors), axis=0)


def binarize(data: BinarizeInput) ->     np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in np.asarray(matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(column).tolist() for column in np.asarray(matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    matrix = np.random.default_rng(seed).normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(statistics: MatrixStatistics):
    """Гистограммы значений по строкам и столбцам матрицы."""
    import matplotlib.pyplot as plt

    matrix = statistics.matrix
    figure, (row_ax, column_ax) = plt.subplots(1, 2, figsize=(12, 4))
    for index, row in enumerate(matrix):
        row_ax.hist(row, alpha=0.5, label=f"Строка {index + 1}")
    for index, column in enumerate(matrix.T):
        column_ax.hist(column, alpha=0.5, label=f"Столбец {index + 1}")
    row_ax.set_title("Значения по строкам")
    column_ax.set_title("Значения по столбцам")
    for axis in (row_ax, column_ax):
        axis.set_xlabel("Значение")
        axis.set_ylabel("Частота")
    if matrix.shape[0] <= 5:
        row_ax.legend()
    if matrix.shape[1] <= 5:
        column_ax.legend()
    figure.tight_layout()
    return figure


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_indices, column_indices = np.indices((rows, columns))
    return np.where((row_indices + column_indices) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    left = (image_width - width) // 2
    top = (image_height - height) // 2
    image[max(top, 0):min(top + height, image_height),
          max(left, 0):min(left + width, image_width)] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    center_x = (image_width - 1) / 2
    center_y = (image_height - 1) / 2
    mask = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values)
    if not 1 <= window <= len(values):
        raise ValueError("Размер окна должен быть от 1 до длины ряда")
    middle = values[1:-1]
    left = values[:-2]
    right = values[2:]
    maxima = np.flatnonzero((middle > left) & (middle > right)) + 1
    minima = np.flatnonzero((middle < left) & (middle < right)) + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        if labels.size == 0:
            class_count = 0
        else:
            class_count = int(labels.max()) + 1
    if np.any(labels < 0) or np.any(labels >= class_count) or class_count < 0:
        raise ValueError("Метки классов должны быть от 0 до class_count - 1")
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
