from satisfy import utils


def test_infinity_is_infinity():
    assert utils.INFINITY is utils.Infinity()


def test_undefined_is_undefined():
    assert utils.UNDEFINED is utils.Undefined()


def test_infinity_is_not_undefined():
    assert utils.INFINITY is not utils.UNDEFINED

