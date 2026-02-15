import cv2
import numpy as np
import pytest
from pathlib import Path

from video2csv.models import ROI, ValueType


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 640x480 BGR frame filled with gray (128)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def sample_rois() -> list[ROI]:
    """Two ROIs within a 640x480 frame."""
    return [
        ROI(name="rpm", x=10, y=10, width=100, height=30, value_type=ValueType.INT),
        ROI(name="afr", x=200, y=10, width=80, height=30, value_type=ValueType.FLOAT),
    ]


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """Create a 5-frame 64x64 video file for testing."""
    path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 64))
    for i in range(5):
        frame = np.full((64, 64, 3), i * 50, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
