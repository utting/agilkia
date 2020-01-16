# -*- coding:utf8

from typing import Optional
import random


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
