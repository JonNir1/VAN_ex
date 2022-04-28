import cv2


class KeyPoint:
    """ A serializable implementation of cv2.KeyPoint objects """

    def __init__(self, x: float, y: float, angle: float, size: float):
        self._x = x
        self._y = y
        self._angle = angle
        self._size = size

    @staticmethod
    def from_cv2_keypoint(kp: cv2.KeyPoint):
        return KeyPoint(x=kp.pt[0], y=kp.pt[1], angle=kp.angle, size=kp.size)

    @property
    def pt(self) -> tuple[float, float]:
        return self._x, self._y

    def __str__(self):
        return str(self.pt)

    def __eq__(self, other):
        if not isinstance(other, KeyPoint):
            return False
        return self._x == other._x and self._y == other._y and self._angle == other._angle and self._size == other._size

    def __hash__(self):
        return hash((self._x, self._y, self._angle, self._size))








