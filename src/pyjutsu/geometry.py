"""
Library of geometry classes.
"""

import dataclasses

from pydantic import BaseModel


class Rect(BaseModel):
    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def from_xywh(x, y, w, h):
        return Rect(x=x, y=y, width=w, height=h)

    def to_vec(self):
        return self.x, self.y, self.width, self.height


class Point(BaseModel):
    x: int
    y: int

    @staticmethod
    def from_xy(x, y):
        return Point(x=x, y=y)

    def to_vec(self):
        return self.x, self.y
