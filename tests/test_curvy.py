import pytest
from matplotlib import pyplot as plt

from curvy import Curvy


@pytest.fixture
def curvy():
    coords = {"Mannheim": (8.03, 49.28, 8.89, 49.72)}

    return Curvy(*coords["Mannheim"], desired_railway_types=["tram", "light_rail"])


def test_curvy(curvy):
    assert True


def test_curvature_plot(curvy):
    line = curvy.railway_lines[0]
    lower, upper = line.get_error_bounds()

    fig, ax = plt.subplots(1, 1)

    ax.plot(line.s, line.c)
    ax.plot(line.s, lower)
    ax.plot(line.s, upper)

    ax.set_xlabel("Distances s [m]")
    ax.set_ylabel("Curvature c [m]")
    ax.grid()

    plt.suptitle(line.name)
    plt.show()
    assert True
