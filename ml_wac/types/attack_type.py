from enum import Enum


class AttackType(Enum):
    RFI = 0
    SQLI = 1
    XSS = 2
    UNKNOWN = 3
    LFI = 4
