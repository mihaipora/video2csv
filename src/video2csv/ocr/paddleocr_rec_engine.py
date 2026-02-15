import logging
import os

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PaddleOCRRecEngine:
    """Recognition-only engine using PaddleOCR's lightweight mobile model.

    Skips text detection entirely — expects a cropped ROI containing
    a single text line (number).  Uses en_PP-OCRv4_mobile_rec (6.8 MB)
    instead of the full pipeline's 84 MB server detection model.
    """

    def __init__(self) -> None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        import warnings
        warnings.filterwarnings("ignore", message=".*ccache.*")

        import logging as _logging
        for name in ["ppocr", "paddle", "paddlex", "paddleocr"]:
            _logging.getLogger(name).setLevel(_logging.ERROR)

        from paddleocr import TextRecognition

        self._rec = TextRecognition(
            model_name="en_PP-OCRv4_mobile_rec",
        )

    def image_to_string(self, image: NDArray[np.uint8]) -> str:
        results = self._rec.predict(input=image, batch_size=1)

        if not results:
            return ""

        res = results[0]
        text = res["rec_text"]
        score = res["rec_score"]

        logger.debug(
            "PaddleOCR rec-only: %r (score=%.3f)",
            text, score,
        )
        return text
