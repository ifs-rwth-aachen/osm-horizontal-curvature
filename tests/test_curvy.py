from curvy import Curvy


def test_curvy():
    coords = {"Mannheim": (8.03, 49.28, 8.89, 49.72)}

    curvy = Curvy(*coords["Mannheim"], desired_railway_types=["tram", "light_rail"])
    assert True