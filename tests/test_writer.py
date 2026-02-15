import csv
from pathlib import Path

from video2csv.models import ROI, ValueType, FrameResult
from video2csv.writer import CSVWriter


def _make_rois():
    return [
        ROI(name="boost_pressure (mbar)", x=0, y=0, width=10, height=10, value_type=ValueType.INT),
        ROI(name="boost_pressure_ref (mbar)", x=0, y=0, width=10, height=10, value_type=ValueType.INT),
    ]


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


class TestCSVWriter:
    def test_writes_header(self, tmp_path: Path):
        out = tmp_path / "test.csv"
        rois = _make_rois()
        with CSVWriter(out, rois):
            pass
        rows = _read_csv(out)
        assert len(rows) == 0
        with open(out) as f:
            header = f.readline().strip()
        assert header == "frame,timestamp_ms,boost_pressure (mbar),boost_pressure_ref (mbar)"

    def test_writes_row(self, tmp_path: Path):
        out = tmp_path / "test.csv"
        rois = _make_rois()
        result = FrameResult(
            frame_index=0,
            timestamp_ms=0.0,
            values={"boost_pressure (mbar)": 2040, "boost_pressure_ref (mbar)": 2017},
        )
        with CSVWriter(out, rois) as w:
            written = w.write_row(result)
        assert written is True
        rows = _read_csv(out)
        assert len(rows) == 1
        assert rows[0]["frame"] == "0"
        assert rows[0]["timestamp_ms"] == "0.0"
        assert rows[0]["boost_pressure (mbar)"] == "2040"
        assert rows[0]["boost_pressure_ref (mbar)"] == "2017"

    def test_skips_row_with_none(self, tmp_path: Path):
        out = tmp_path / "test.csv"
        rois = _make_rois()
        result = FrameResult(
            frame_index=0,
            timestamp_ms=0.0,
            values={"boost_pressure (mbar)": 2040, "boost_pressure_ref (mbar)": None},
        )
        with CSVWriter(out, rois) as w:
            written = w.write_row(result)
        assert written is False
        rows = _read_csv(out)
        assert len(rows) == 0

    def test_writes_row_with_none_when_skip_disabled(self, tmp_path: Path):
        out = tmp_path / "test.csv"
        rois = _make_rois()
        result = FrameResult(
            frame_index=0,
            timestamp_ms=0.0,
            values={"boost_pressure (mbar)": 2040, "boost_pressure_ref (mbar)": None},
        )
        with CSVWriter(out, rois) as w:
            written = w.write_row(result, skip_on_any_none=False)
        assert written is True
        rows = _read_csv(out)
        assert len(rows) == 1
        assert rows[0]["boost_pressure_ref (mbar)"] == ""

    def test_multiple_rows(self, tmp_path: Path):
        out = tmp_path / "test.csv"
        rois = _make_rois()
        with CSVWriter(out, rois) as w:
            for i in range(5):
                result = FrameResult(
                    frame_index=i,
                    timestamp_ms=i * 33.3,
                    values={"boost_pressure (mbar)": 1000 + i, "boost_pressure_ref (mbar)": 2000 + i},
                )
                w.write_row(result)
        rows = _read_csv(out)
        assert len(rows) == 5
        assert rows[4]["frame"] == "4"

    def test_timestamp_precision(self, tmp_path: Path):
        out = tmp_path / "test.csv"
        rois = _make_rois()
        result = FrameResult(
            frame_index=0,
            timestamp_ms=33.33333,
            values={"boost_pressure (mbar)": 100, "boost_pressure_ref (mbar)": 200},
        )
        with CSVWriter(out, rois) as w:
            w.write_row(result)
        rows = _read_csv(out)
        assert rows[0]["timestamp_ms"] == "33.3"
