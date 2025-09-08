#priority.py

from enum import IntEnum


class Priority(IntEnum):
    IMMEDIATE = 0
    HIGH = 1
    MEDIUM = 2
    NORMAL = 3
    SYSTEM_SLOW = 4