from video2csv.ocr.base import OCREngine
from video2csv.ocr.paddleocr_engine import PaddleOCREngine
from video2csv.ocr.paddleocr_mobile_engine import PaddleOCRMobileEngine
from video2csv.ocr.paddleocr_rec_engine import PaddleOCRRecEngine

_REGISTRY: dict[str, type] = {
    "paddleocr": PaddleOCREngine,
    "paddleocr-mobile": PaddleOCRMobileEngine,
    "paddleocr-rec": PaddleOCRRecEngine,
}


def create_engine(name: str, **kwargs) -> OCREngine:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown OCR engine '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[name]
    engine = cls(**kwargs)
    assert isinstance(engine, OCREngine), (
        f"{cls.__name__} does not satisfy OCREngine protocol"
    )
    return engine


def register_engine(name: str, cls: type) -> None:
    _REGISTRY[name] = cls
