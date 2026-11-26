"""
Unit tests for exercise8.py: mini_jpeg_pipeline, padding, PSNR, FFmpeg baseline.
Run with: pytest -v tests/test_mini_jpeg_pipeline.py
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from exercise8 import (
    compute_psnr,
    pad_image_to_8x8,
    mini_jpeg_pipeline,
    ffmpeg_jpeg_baseline,
)
from first_seminar import ColorTranslator


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def translator() -> ColorTranslator:
    return ColorTranslator()


@pytest.fixture
def y_channel_16x16() -> np.ndarray:
    """16x16 synthetic Y channel with gradient."""
    return np.tile(np.arange(256, dtype=np.uint8).reshape(16, 16), (1, 1))


@pytest.fixture
def y_channel_10x10() -> np.ndarray:
    """10x10 synthetic Y channel (not multiple of 8)."""
    return np.tile(np.arange(100, dtype=np.uint8).reshape(10, 10), (1, 1))


@pytest.fixture
def y_channel_8x8() -> np.ndarray:
    """8x8 block from exercise6."""
    return np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ], dtype=np.uint8)


# ============================================================
# Padding Tests
# ============================================================
class TestPadImageTo8x8:
    def test_no_pad_needed(self, y_channel_16x16):
        """16x16 (multiple of 8) -> no padding."""
        padded, orig_h, orig_w = pad_image_to_8x8(y_channel_16x16)
        assert padded.shape == (16, 16)
        assert orig_h == 16
        assert orig_w == 16
        np.testing.assert_array_equal(padded, y_channel_16x16)

    def test_pad_needed(self, y_channel_10x10):
        """10x10 -> padded to 16x16 with edge values."""
        padded, orig_h, orig_w = pad_image_to_8x8(y_channel_10x10)
        assert padded.shape == (16, 16)
        assert orig_h == 10
        assert orig_w == 10
        # Top-left 10x10 should match original
        np.testing.assert_array_equal(padded[:10, :10], y_channel_10x10)
        # Padded region should be edge-replicated (mode="edge" replicates last row/col)
        # Last row (index 9) replicated to rows 10-15 across full width (16)
        assert np.all(padded[10:, :] == padded[9:10, :])
        # Last col (index 9) replicated to cols 10-15 across full height (16)
        assert np.all(padded[:, 10:] == padded[:, 9:10])

    def test_pad_exact_multiple(self):
        """8x8 -> no change."""
        img = np.arange(64, dtype=np.uint8).reshape(8, 8)
        padded, h, w = pad_image_to_8x8(img)
        assert padded.shape == (8, 8)
        np.testing.assert_array_equal(padded, img)


# ============================================================
# PSNR Tests
# ============================================================
class TestComputePSNR:
    def test_identical_images_inf_psnr(self, y_channel_16x16):
        """Identical images -> PSNR = inf."""
        psnr = compute_psnr(y_channel_16x16, y_channel_16x16)
        assert psnr == float("inf")

    def test_known_mse_psnr(self):
        """Known MSE -> expected PSNR."""
        img1 = np.full((8, 8), 100, dtype=np.uint8)
        img2 = np.full((8, 8), 110, dtype=np.uint8)
        # MSE = (10^2) = 100
        # PSNR = 20 * log10(255 / sqrt(100)) = 20 * log10(25.5) ≈ 28.13 dB
        psnr = compute_psnr(img1, img2)
        assert abs(psnr - 28.13) < 0.05

    def test_psnr_symmetry(self, y_channel_16x16):
        """PSNR(img1, img2) == PSNR(img2, img1)."""
        img2 = y_channel_16x16.copy()
        img2[0, 0] = 200
        psnr1 = compute_psnr(y_channel_16x16, img2)
        psnr2 = compute_psnr(img2, y_channel_16x16)
        assert abs(psnr1 - psnr2) < 1e-10


# ============================================================
# Mini-JPEG Pipeline Tests
# ============================================================
class TestMiniJPEGPipeline:
    def test_pipeline_roundtrip_qf90(self, translator, y_channel_16x16):
        """Full pipeline at QF=90: high quality, PSNR > 35 dB."""
        rec, rle_symbols = mini_jpeg_pipeline(y_channel_16x16, 90, translator)
        assert rec.shape == y_channel_16x16.shape
        assert rec.dtype == np.uint8
        psnr = compute_psnr(y_channel_16x16, rec)
        assert psnr > 35, f"QF=90 PSNR={psnr:.2f} dB (expected > 35)"
        assert rle_symbols > 0

    def test_pipeline_roundtrip_qf50(self, translator, y_channel_16x16):
        """Full pipeline at QF=50: medium quality, PSNR > 25 dB."""
        rec, rle_symbols = mini_jpeg_pipeline(y_channel_16x16, 50, translator)
        psnr = compute_psnr(y_channel_16x16, rec)
        assert psnr > 25, f"QF=50 PSNR={psnr:.2f} dB (expected > 25)"

    def test_pipeline_roundtrip_qf10(self, translator, y_channel_16x16):
        """Full pipeline at QF=10: low quality, PSNR > 15 dB, many zeros."""
        rec, rle_symbols = mini_jpeg_pipeline(y_channel_16x16, 10, translator)
        psnr = compute_psnr(y_channel_16x16, rec)
        assert psnr > 15, f"QF=10 PSNR={psnr:.2f} dB (expected > 15)"
        # At low QF, RLE should be more compact (more zeros)
        assert rle_symbols < 4 * 64  # 4 blocks * 64 max symbols

    def test_pipeline_non_multiple_of_8(self, translator, y_channel_10x10):
        """10x10 image -> pad -> decode -> crop = original dims."""
        rec, _ = mini_jpeg_pipeline(y_channel_10x10, 50, translator)
        assert rec.shape == (10, 10)
        assert rec.dtype == np.uint8

    def test_pipeline_8x8_single_block(self, translator, y_channel_8x8):
        """8x8 single block processes correctly."""
        rec, _ = mini_jpeg_pipeline(y_channel_8x8, 50, translator)
        assert rec.shape == (8, 8)
        psnr = compute_psnr(y_channel_8x8, rec)
        assert psnr > 20  # Single block, moderate QF


# ============================================================
# FFmpeg Baseline Tests (Mocked)
# ============================================================
class TestFFmpegJPEGBaseline:
    @patch("exercise8.subprocess.run")
    @patch("exercise8.Image.open")
    @patch("exercise8.os.path.getsize")
    @patch("exercise8.os.remove")
    @patch("exercise8.np.array")
    def test_ffmpeg_jpeg_baseline_mocked(
        self, mock_np_array, mock_remove, mock_getsize, mock_image_open, mock_run
    ):
        """ffmpeg_jpeg_baseline calls ffmpeg and returns decoded image + size."""
        # Setup mocks
        mock_run.return_value = MagicMock(returncode=0)
        mock_getsize.return_value = 1234
        
        # Mock PIL Image.open context manager
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img
        
        # Mock np.array to return test array
        mock_np_array.return_value = np.zeros((16, 16), dtype=np.uint8)
        
        # Call function
        decoded_img, file_size = ffmpeg_jpeg_baseline("dummy_input.png", 10)
        
        # Assertions
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert isinstance(cmd, str) or "ffmpeg" in str(cmd)
        assert "dummy_input.png" in str(cmd)
        assert "-q:v" in str(cmd) or "10" in str(cmd)
        assert file_size == 1234
        assert decoded_img.shape == (16, 16)
        mock_getsize.assert_called_once()
        mock_remove.assert_called_once()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])