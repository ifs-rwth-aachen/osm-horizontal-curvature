import numpy as np


def poly_quadratic(x, a, b):
    return a * x ** 2 + b * x


def get_error_bounds(curvature_array):
    if isinstance(curvature_array, np.array):
        error = poly_quadratic(curvature_array, 2.24, 0.15)
        upper = curvature_array + error
        lower = curvature_array - error
        return lower, upper

    else:
        raise TypeError(f'Your input is not of type np.array --> {type(curvature_array)}')
