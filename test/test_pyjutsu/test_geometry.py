from pyjutsu.geometry import Rect, Point


def test_rect():
    r = Rect.from_xywh(0, 100, 200, 300)
    x, y, w, h = r.to_vec()
    assert x == 0
    assert y == 100
    assert w == 200
    assert h == 300


def test_point():
    p = Point.from_xy(0, 100)
    x, y = p.to_vec()
    assert x == 0
    assert y == 100
