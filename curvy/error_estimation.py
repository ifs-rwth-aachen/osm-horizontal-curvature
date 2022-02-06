import numpy as np


def get_error_bounds(curvature_array):
    error = poly_quadratic(curvature_array, 2.24, 0.15)
    upper = curvature_array + error
    lower = curvature_array - error
    return lower, upper
