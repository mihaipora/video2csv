import numpy as np
import pytest

from video2csv.ocr.base import OCREngine
from video2csv.ocr.factory import create_engine, register_engine


class FakeEngine:
    """A minimal class satisfying OCREngine protocol."""

    def image_to_string(self, image: np.ndarray) -> str:
        return "42"


class TestOCRProtocol:
    def test_fake_engine_satisfies_protocol(self):
        assert isinstance(FakeEngine(), OCREngine)

    def test_object_does_not_satisfy_protocol(self):
        assert not isinstance(object(), OCREngine)


class TestFactory:
    def test_create_paddleocr(self):
        engine = create_engine("paddleocr")
        assert isinstance(engine, OCREngine)

    def test_unknown_engine_raises(self):
        with pytest.raises(KeyError, match="Unknown OCR engine"):
            create_engine("nonexistent")

    def test_register_custom_engine(self):
        register_engine("fake", FakeEngine)
        engine = create_engine("fake")
        assert engine.image_to_string(np.zeros((10, 10), dtype=np.uint8)) == "42"
