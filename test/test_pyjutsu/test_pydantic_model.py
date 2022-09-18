import numpy as np

from pyjutsu.pydantic_model import PydanticModel, NDArray


def test_pydantic_model():
    class X(PydanticModel):
        v1: int
        a1: NDArray

    x = X(
        v1=123,
        a1=np.array([1, 2, 3])
    )

    x_json = x.json()

    assert (
        x_json == '{"v1": 123, "a1": {'
                  '"ndarray_base64": "AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA",'
                  ' "array_dtype": "int64",'
                  ' "array_shape": [3]'
                  '}}'
    )

    x2 = X(
        v1=123,
        a1=np.array([1, 2, 3])
    )

    assert x == x2

    y = X(
        v1=123,
        a1=np.array([1, 2, 3, 4])
    )

    assert x != y
