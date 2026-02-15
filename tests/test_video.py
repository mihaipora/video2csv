from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from video2csv.video import open_video, iter_frames, read_frame_at, frame_timestamp_ms


class TestFrameTimestampMs:
    def test_first_frame(self):
        assert frame_timestamp_ms(0, 30.0) == 0.0

    def test_frame_at_one_second(self):
        assert frame_timestamp_ms(30, 30.0) == 1000.0

    def test_frame_at_half_second(self):
        assert frame_timestamp_ms(15, 30.0) == 500.0

    def test_60fps(self):
        assert frame_timestamp_ms(60, 60.0) == 1000.0

    def test_fractional_fps(self):
        result = frame_timestamp_ms(1, 29.97)
        assert abs(result - 33.3667) < 0.01


class TestOpenVideo:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            open_video(tmp_path / "nonexistent.mp4")

    def test_opens_valid_video(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        try:
            assert meta.width == 64
            assert meta.height == 64
            assert meta.fps == 30.0
            assert meta.total_frames == 5
            assert abs(meta.duration_ms - (5 / 30.0 * 1000.0)) < 0.1
        finally:
            cap.release()


class TestIterFrames:
    def test_yields_all_frames(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        frames = list(iter_frames(cap, meta.total_frames))
        assert len(frames) == 5

    def test_frame_indices_sequential(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        indices = [idx for idx, _ in iter_frames(cap, meta.total_frames)]
        assert indices == [0, 1, 2, 3, 4]

    def test_frames_are_numpy_arrays(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        for _, frame in iter_frames(cap, meta.total_frames):
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (64, 64, 3)
            assert frame.dtype == np.uint8

    def test_releases_capture_on_completion(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        list(iter_frames(cap, meta.total_frames))
        # After iteration, capture should be released
        assert not cap.isOpened()


class TestReadFrameAt:
    def test_reads_first_frame(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        try:
            frame = read_frame_at(cap, 0)
            assert frame is not None
            assert frame.shape == (64, 64, 3)
        finally:
            cap.release()

    def test_reads_last_frame(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        try:
            frame = read_frame_at(cap, meta.total_frames - 1)
            assert frame is not None
        finally:
            cap.release()

    def test_returns_none_past_end(self, tiny_video: Path):
        cap, meta = open_video(tiny_video)
        try:
            frame = read_frame_at(cap, meta.total_frames + 100)
            assert frame is None
        finally:
            cap.release()
