import base64
from typing import Any, List

import numpy as np
from pydantic.main import BaseModel


class NDArraySerialized(BaseModel):
    ndarray_base64: str
    array_dtype: str
    array_shape: List[int]


def _numpy_encoder(npa: np.ndarray):
    assert isinstance(npa, np.ndarray)
    data_b64 = base64.b64encode(npa.data)
    return NDArraySerialized(
        ndarray_base64=data_b64.decode("ascii"),
        array_dtype=str(npa.dtype),
        array_shape=npa.shape
    ).dict()


def _numpy_decoder(npa_serialized: NDArraySerialized):

    def int_or_str(n):
        try:
            return int(n)
        except ValueError:
            return n

    data_b64 = npa_serialized.ndarray_base64
    if data_b64:
        data = base64.b64decode(data_b64.encode())
        return np.frombuffer(data, npa_serialized.array_dtype).reshape(
            npa_serialized.array_shape
        )


class NDArray(np.ndarray):
    """
    Partial UK postcode validation. Note: this is just an example, and is not
    intended for use in production; in particular this does NOT guarantee
    a postcode exists, just that it has a valid format.
    """
    def __init__(self, *v, **kw):
        super().__init__(*v, **kw)

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(NDArraySerialized.schema())

    @classmethod
    def validate(cls, v):
        if isinstance(v, np.ndarray):
            return v
        res = NDArraySerialized.validate(v)
        return cls(_numpy_decoder(res))



    def __repr__(self):
        return super().__repr__()

    # TODO: I also want to override dict() behaviour, but there
    #    is no way to do it so far.


class PydanticModel(BaseModel):
    class Config:
        json_encoders = {
            np.ndarray: _numpy_encoder
        }

    def __eq__(self, other):

        left_type = type(self)
        right_type = type(other)

        if left_type is not right_type:
            return super().__eq__(other)

        self_fields = self.dict()
        other_fields = other.dict()

        if set(self_fields.keys()).difference(other_fields.keys()):
            return False

        return all([
            (v == vo).all()
            if (
                isinstance(v, np.ndarray) and isinstance(vo, np.ndarray)
                and v.shape == vo.shape
            ) else v == vo
            for (k, v), (ko, vo) in zip(self_fields.items(), other_fields.items())
        ])
