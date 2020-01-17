# -*- coding:utf8

from typing import Optional
import random


class MinList:
    def __init__(self, size):
        """Create a list of minimals values and the indexes associated with thoses values
        :para size: the number of values to keep
        """
        assert size > 0
        self.values = []
        self.indexes = []
        self.size = size

    def insert(self, value, index=None):
        """add a new value and its index in the list
        :param value: the value to keep if beeing in the smallest of the list
        :param index: the index associated to this value
        """
        if len(self.values) == 0:
            self.values.append(value)
            self.indexes.append(index)
            return
        last = len(self.values) - 1
        if value > self.values[last] and len(self.values) >= self.size:
            return
        if len(self.values) < self.size:
            self.values.append(value)
            self.indexes.append(index)
        pos = len(self.values) - 2
        while pos >= 0 and self.values[pos] > value:
            self.values[pos + 1] = self.values[pos]
            self.indexes[pos + 1] = self.indexes[pos]
            pos -= 1
        self.values[pos + 1] = value
        self.indexes[pos + 1] = index

    def pick(self):
        """remove the minimal value and its associated index from the list and return them
        """
        if len(self.values) == 0:
            return
        value = self.values[0]
        index = self.indexes[0]
        self.values = self.values[1:]
        self.indexes = self.indexes[1:]
        return value, index


class Color:
    def __init__(self, r: int, g: int, b: int, alpha: Optional[int] = None):
        """
        Create a new color. `r`, `g`, `b` and `alpha` values must be between 0 and 255 (`alpha` can also be `None`).
        """
        assert 0 <= r and r <= 255 and 0 <= g and g <= 255 and 0 <= b and b <= 255 and (
            alpha is None or (0 <= alpha and alpha <= 255))
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.alpha = None if alpha is None else int(alpha)

    def average(self, other: 'Color', otherWeight: float) -> 'Color':
        """
        Create a new color between this color and the `other`.

        Params:
            other: the other color used to deviate from self.
            otherWeight: a float between 0 and 1 used to indicate the ponderation of `other` color.
                `0.` will return the same color as self, `1.` will return the same color as `other`.
        """
        weight = 1 - otherWeight
        r = int(self.r * weight + otherWeight * other.r)
        g = int(self.g * weight + otherWeight * other.g)
        b = int(self.b * weight + otherWeight * other.b)
        alpha = None
        if self.alpha is not None and other.alpha is not None:
            alpha = int(self.alpha * weight + otherWeight * other.alpha)
        return Color(r, g, b, alpha)

    def toHex(self) -> str:
        """
        export this color in hexadecimal format #RRGGBB(AA)
        """
        base = "#{:02x}{:02x}{:02x}".format(self.r, self.g, self.b)
        if self.alpha is not None:
            base += "{:02x}".format(self.alpha)
        return base


class ColorList:
    """
    This class aims to provide colors which can be used to draw clusters.
    """

    def __init__(self):
        """
        Create a ColorList with some colors already defined.
        """
        self.colors = [
            Color(0xff, 0x00, 0x00),
            Color(0x00, 0xff, 0x00),
            Color(0xff, 0x00, 0xff),
            Color(0xff, 0xa0, 0x00),
            Color(0x00, 0xff, 0xff),
            Color(0xff, 0x88, 0x88),
            Color(0x88, 0xff, 0x88),
            Color(0x88, 0x88, 0xff),
            Color(0xff, 0xf6, 0x00),
            Color(0x00, 0xcc, 0x99),
            Color(0x00, 0x47, 0xab),
            Color(0xff, 0x7e, 0x00),
            Color(0x8a, 0x2b, 0xe2),
            Color(0x25, 0xde, 0xbc),
            Color(0x86, 0x3b, 0x7d),
            Color(0xcc, 0x69, 0xde)]

    def pickColor(self) -> Color:
        """
        Get another color.

        This method create random colors if needed but never returns None.
        """
        if len(self.colors) == 0:
            return Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = self.colors[0]
        self.colors = self.colors[1:]
        return color
