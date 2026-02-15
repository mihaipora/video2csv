from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class OCREngine(Protocol):
    def image_to_string(self, image: NDArray[np.uint8]) -> str: ...
