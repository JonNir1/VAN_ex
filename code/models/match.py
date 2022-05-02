import cv2
import numpy as np
# from abc import ABCMeta, ABC, abstractmethod
from typing import NamedTuple

from models.directions import Side, Position
from models.keypoint import KeyPoint

# TODO: make this work with IMatch as well
# class NT(NamedTuple):
#     # fake class to enable multiple inheritance when using NamedTuple
#     # see: https://stackoverflow.com/questions/51860186/namedtuple-class-with-abc-mixin
#     pass
#
#
# class IMatch(ABCMeta):
#
#     @classmethod
#     def __subclasscheck__(cls, subclass):
#         get_keypoint_attr = getattr(subclass, 'get_keypoint', __default=None)
#         get_pixels_attr = getattr(subclass, 'get_pixels', __default=None)
#         if get_keypoint_attr is None:
#             return False
#         if get_pixels_attr is None:
#             return False
#         return callable(get_keypoint_attr) and callable(get_pixels_attr)


class FrameMatch(NamedTuple):

    left_keypoint: cv2.KeyPoint
    right_keypoint: cv2.KeyPoint

    def get_keypoint(self, s: Side) -> cv2.KeyPoint:
        if s == Side.LEFT:
            return self.left_keypoint
        return self.right_keypoint

    def get_pixels(self) -> np.ndarray:
        # returns a 2x2 array where each row is a keypoint and each col is a coordinate (X, Y)
        return np.array([self.left_keypoint.pt, self.right_keypoint.pt])

    def __eq__(self, other):
        if not isinstance(other, FrameMatch):
            return False
        return self.left_keypoint == other.left_keypoint and self.right_keypoint == other.right_keypoint

    def __hash__(self):
        return hash(id(self))


class MutualMatch(NamedTuple):

    back_frame_match: FrameMatch
    front_frame_match: FrameMatch

    def get_frame_match(self, p: Position) -> FrameMatch:
        if p == Position.BACK:
            return self.back_frame_match
        return self.front_frame_match

    def get_keypoint(self, s: Side, p: Position) -> cv2.KeyPoint:
        frame_match = self.get_frame_match(p)
        return frame_match.get_keypoint(s)

    def get_pixels(self) -> np.ndarray:
        # returns a 4x2 array where each row is a keypoint and each col is a coordinate (X, Y)
        back_pixels = self.get_frame_match(Position.BACK).get_pixels()
        front_pixels = self.get_frame_match(Position.FRONT).get_pixels()
        return np.vstack([back_pixels, front_pixels])

    def __eq__(self, other):
        if not isinstance(other, MutualMatch):
            return False
        return self.back_frame_match == other.back_frame_match and self.front_frame_match == other.front_frame_match

    def __str__(self):
        return f"MM({str(self.back_frame_match)} | {str(self.front_frame_match)})"

    def __hash__(self):
        return hash(id(self))
