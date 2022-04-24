from enum import Enum, unique


@unique
class Side(Enum):
    LEFT = 'L'
    RIGHT = 'R'


@unique
class Position(Enum):
    BACK = 0
    FRONT = 1

