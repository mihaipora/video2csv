from dataclasses import dataclass
from enum import Enum


class ValueType(Enum):
    INT = "int"
    FLOAT = "float"


@dataclass(frozen=True, slots=True)
class ROI:
    name: str
    x: int
    y: int
    width: int
    height: int
    value_type: ValueType
    signed: bool = False


@dataclass(frozen=True, slots=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: float


@dataclass(slots=True)
class FrameResult:
    frame_index: int
    timestamp_ms: float
    values: dict[str, int | float | None]
