import logging
import os
import re

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _has_digits(text: str) -> bool:
    return bool(re.search(r"\d", text))


class PaddleOCREngine:
    def __init__(self) -> None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        import warnings
        warnings.filterwarnings("ignore", message=".*ccache.*")

        import logging as _logging
        for name in ["ppocr", "paddle", "paddlex", "paddleocr"]:
            _logging.getLogger(name).setLevel(_logging.ERROR)

        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

    def image_to_string(self, image: NDArray[np.uint8]) -> str:
        results = self._ocr.predict(image)

        # Collect all detected text boxes with their area
        candidates = []
        for res in results:
            texts = res["rec_texts"]
            polys = res["rec_polys"]
            scores = res["rec_scores"]

            for text, poly, score in zip(texts, polys, scores):
                xs = poly[:, 0]
                ys = poly[:, 1]
                area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
                candidates.append((text, area, score))

        if not candidates:
            return ""

        # Prefer text containing digits; among those, pick largest area.
        # Fall back to largest area overall if no digits found.
        numeric = [(t, a, s) for t, a, s in candidates if _has_digits(t)]
        pool = numeric if numeric else candidates
        best = max(pool, key=lambda x: x[1])

        logger.debug(
            "PaddleOCR picked: %r (area=%.0f, score=%.3f) from %d candidates",
            best[0], best[1], best[2], len(candidates),
        )
        return best[0]
