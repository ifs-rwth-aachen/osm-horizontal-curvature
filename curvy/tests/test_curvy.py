import pytest

from curvy import Curvy

coords = [[(-1, -1, -1, -1), False],
          [(-1, -1, 90, 90), True],
          [(4, -1, 4, -1), False],
          [(8.03, 49.28, 8.89, 49.72), True]]

railway_types = [["rail", True],
                 ["light_rail", True],
                 ["tram", True],
                 ["subway", True],
                 ["highway", False]]


@pytest.fixture
def curvy():
    return Curvy(*(8.03, 49.28, 8.89, 49.72))


@pytest.mark.parametrize("coord, valid", coords)
def test_curvy_initialization(coord, valid):
    try:
        Curvy(lon_sw=coord[0], lat_sw=coord[1], lon_ne=coord[2], lat_ne=coord[3])
        assert valid
    except ValueError:
        assert True


@pytest.mark.parametrize("type, valid", railway_types)
def test_query_creation(curvy, type, valid):
    try:
        curvy._create_query(railway_type=type)
        assert valid
    except ValueError:
        assert True


def test_data_download(curvy):
    curvy.download_track_data()
    assert True
