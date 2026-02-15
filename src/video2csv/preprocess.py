import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


def crop_roi(frame: Frame, x: int, y: int, width: int, height: int) -> Frame:
    return frame[y : y + height, x : x + width]


def preprocess_roi(
    image: Frame,
    *,
    upscale_factor: int = 3,
    threshold_value: int = 127,
    invert: bool = True,
) -> Frame:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold_value, 255, thresh_type)

    h, w = binary.shape[:2]
    upscaled = cv2.resize(
        binary,
        (w * upscale_factor, h * upscale_factor),
        interpolation=cv2.INTER_CUBIC,
    )
    return upscaled
