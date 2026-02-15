import numpy as np

from video2csv.preprocess import crop_roi, preprocess_roi


class TestCropRoi:
    def test_crops_correct_region(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[10:40, 20:70, :] = 255  # white rectangle
        crop = crop_roi(frame, x=20, y=10, width=50, height=30)
        assert crop.shape == (30, 50, 3)
        assert np.all(crop == 255)

    def test_crop_is_view_not_copy(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_roi(frame, x=0, y=0, width=10, height=10)
        crop[:] = 128
        assert frame[0, 0, 0] == 128


class TestPreprocessRoi:
    def test_output_is_single_channel(self):
        image = np.full((30, 50, 3), 200, dtype=np.uint8)
        result = preprocess_roi(image)
        assert len(result.shape) == 2

    def test_output_is_upscaled(self):
        image = np.full((30, 50, 3), 200, dtype=np.uint8)
        result = preprocess_roi(image, upscale_factor=3)
        assert result.shape == (90, 150)

    def test_output_is_binary(self):
        image = np.full((30, 50, 3), 200, dtype=np.uint8)
        result = preprocess_roi(image)
        unique = set(np.unique(result))
        assert unique <= {0, 255}

    def test_custom_upscale_factor(self):
        image = np.full((20, 40, 3), 100, dtype=np.uint8)
        result = preprocess_roi(image, upscale_factor=2)
        assert result.shape == (40, 80)

    def test_accepts_grayscale_input(self):
        image = np.full((30, 50), 200, dtype=np.uint8)
        result = preprocess_roi(image, upscale_factor=2)
        assert result.shape == (60, 100)

    def test_invert_false(self):
        # White image (200 > 127) with invert=False -> should stay white (255)
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = preprocess_roi(image, upscale_factor=1, invert=False)
        assert np.all(result == 255)

    def test_invert_true(self):
        # White image (200 > 127) with invert=True -> should become black (0)
        image = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = preprocess_roi(image, upscale_factor=1, invert=True)
        assert np.all(result == 0)
