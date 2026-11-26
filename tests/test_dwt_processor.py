"""
Unit tests for exercise9.py: DWTProcessor, FFmpeg PSNR/SSIM, JPEG vs JPEG2000 battle.
Run with: pytest -v tests/test_dwt_processor.py
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from exercise9 import (
    DWTProcessor,
    get_ffmpeg_psnr_ssim,
    run_battle,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def dwt_haar() -> DWTProcessor:
    return DWTProcessor(wavelet="haar")


@pytest.fixture
def dwt_db2() -> DWTProcessor:
    return DWTProcessor(wavelet="db2")


@pytest.fixture
def gray_32x32() -> np.ndarray:
    """32x32 synthetic grayscale image within 0-255 range for perfect reconstruction test."""
    # Use values 0-255 to avoid clipping in decode()
    return np.tile(np.arange(256, dtype=np.float64).reshape(16, 16), (2, 2))


@pytest.fixture
def gray_16x16() -> np.ndarray:
    """16x16 synthetic grayscale image within 0-255 range."""
    return np.tile(np.arange(256, dtype=np.float64).reshape(16, 16), (1, 1))


@pytest.fixture
def gray_8x8() -> np.ndarray:
    """8x8 synthetic grayscale image within 0-255 range."""
    return np.arange(64, dtype=np.float64).reshape(8, 8)


@pytest.fixture
def gray_32x32_full() -> np.ndarray:
    """32x32 synthetic grayscale image with full range (for encode test only)."""
    return np.tile(np.arange(1024, dtype=np.float64).reshape(32, 32), (1, 1))


# ============================================================
# DWTProcessor Tests
# ============================================================
class TestDWTProcessorEncode:
    def test_encode_returns_four_subbands(self, dwt_haar, gray_32x32_full):
        """encode returns LL, LH, HL, HH each half size."""
        LL, LH, HL, HH = dwt_haar.encode(gray_32x32_full)
        assert LL.shape == (16, 16)
        assert LH.shape == (16, 16)
        assert HL.shape == (16, 16)
        assert HH.shape == (16, 16)

    def test_encode_16x16(self, dwt_haar, gray_16x16):
        """encode on 16x16 returns 8x8 subbands."""
        LL, LH, HL, HH = dwt_haar.encode(gray_16x16)
        assert LL.shape == (8, 8)
        assert LH.shape == (8, 8)

    def test_encode_different_wavelets(self, gray_32x32_full):
        """Different wavelets produce different subband values (at least for LL)."""
        dwt_haar = DWTProcessor(wavelet="haar")
        dwt_db2 = DWTProcessor(wavelet="db2")
        LL1, _, _, _ = dwt_haar.encode(gray_32x32_full)
        LL2, _, _, _ = dwt_db2.encode(gray_32x32_full)
        # Different wavelets -> different approximations (shapes may differ due to border handling)
        # Just verify they're not identical where comparable
        min_shape = (min(LL1.shape[0], LL2.shape[0]), min(LL1.shape[1], LL2.shape[1]))
        assert not np.allclose(LL1[:min_shape[0], :min_shape[1]], LL2[:min_shape[0], :min_shape[1]])


class TestDWTProcessorDecode:
    def test_decode_perfect_reconstruction_haar(self, dwt_haar, gray_32x32):
        """encode -> decode = original (Haar is orthogonal, perfect reconstruction within 0-255)."""
        LL, LH, HL, HH = dwt_haar.encode(gray_32x32)
        reconstructed = dwt_haar.decode(LL, LH, HL, HH)
        # Haar DWT has perfect reconstruction (up to numerical precision) for values in 0-255
        max_error = np.max(np.abs(gray_32x32 - reconstructed))
        assert max_error < 1e-10, f"Max reconstruction error: {max_error}"

    def test_decode_perfect_reconstruction_db2(self, dwt_db2, gray_32x32):
        """db2 wavelet also has perfect reconstruction for values in 0-255."""
        LL, LH, HL, HH = dwt_db2.encode(gray_32x32)
        reconstructed = dwt_db2.decode(LL, LH, HL, HH)
        max_error = np.max(np.abs(gray_32x32 - reconstructed))
        assert max_error < 1e-10

    def test_decode_wrong_wavelet_produces_wrong_result(self, dwt_haar, gray_32x32):
        """decode with mismatched wavelet produces incorrect reconstruction (no exception raised).
        
        Note: db2 has longer filter so output shape may differ. We compare overlapping region.
        """
        LL, LH, HL, HH = dwt_haar.encode(gray_32x32)
        dwt_db2 = DWTProcessor(wavelet="db2")
        reconstructed = dwt_db2.decode(LL, LH, HL, HH)
        # Compare overlapping region (db2 output may be smaller due to longer filter)
        min_h = min(gray_32x32.shape[0], reconstructed.shape[0])
        min_w = min(gray_32x32.shape[1], reconstructed.shape[1])
        max_error = np.max(np.abs(gray_32x32[:min_h, :min_w] - reconstructed[:min_h, :min_w]))
        # Should NOT match original (mismatched wavelet)
        assert max_error > 1.0, "Mismatched wavelet should produce wrong reconstruction"

    def test_decode_clips_to_uint8(self, dwt_haar):
        """decode clips and rounds to uint8 [0, 255]."""
        # Create subbands that would reconstruct outside 0-255
        LL = np.full((16, 16), 300.0)
        LH = np.zeros((16, 16))
        HL = np.zeros((16, 16))
        HH = np.zeros((16, 16))
        reconstructed = dwt_haar.decode(LL, LH, HL, HH)
        assert reconstructed.dtype == np.uint8
        assert np.all(reconstructed >= 0)
        assert np.all(reconstructed <= 255)


class TestDWTProcessorSaveSubbandsGrid:
    @patch("exercise9.Image.fromarray")
    def test_save_subbands_grid_calls_save(self, mock_fromarray, dwt_haar, gray_32x32):
        """save_subbands_grid creates grid and saves via PIL."""
        mock_img = MagicMock()
        mock_fromarray.return_value = mock_img
        
        LL, LH, HL, HH = dwt_haar.encode(gray_32x32)
        dwt_haar.save_subbands_grid(LL, LH, HL, HH, "test_output.png")
        
        mock_fromarray.assert_called_once()
        mock_img.save.assert_called_once_with("test_output.png")


# ============================================================
# FFmpeg PSNR/SSIM Parsing Tests (Mocked)
# ============================================================
class TestGetFFmpegPSNRSSIM:
    @patch("exercise9.subprocess.run")
    def test_parses_psnr_ssim_from_stderr(self, mock_run):
        """get_ffmpeg_psnr_ssim parses average PSNR and All SSIM from FFmpeg stderr."""
        mock_result = MagicMock()
        mock_result.stderr = """
        [Parsed_psnr_0 @ 0x...] PSNR y:30.12 u:31.45 v:32.78 average:31.45
        [Parsed_ssim_0 @ 0x...] SSIM y:0.9234 u:0.9345 v:0.9456 All:0.9345
        """
        mock_run.return_value = mock_result
        
        psnr, ssim = get_ffmpeg_psnr_ssim("ref.png", "dist.png")
        
        assert psnr == 31.45
        assert ssim == 0.9345
        mock_run.assert_called_once()

    @patch("exercise9.subprocess.run")
    def test_handles_missing_values(self, mock_run):
        """Returns 0.0 if parsing fails."""
        mock_result = MagicMock()
        mock_result.stderr = "No PSNR or SSIM here"
        mock_run.return_value = mock_result
        
        psnr, ssim = get_ffmpeg_psnr_ssim("ref.png", "dist.png")
        
        assert psnr == 0.0
        assert ssim == 0.0


# ============================================================
# JPEG vs JPEG2000 Battle Tests (Mocked)
# ============================================================
class TestRunBattle:
    @patch("exercise9.subprocess.run")
    @patch("exercise9.Image.open")
    @patch("exercise9.os.path.getsize")
    @patch("exercise9.get_ffmpeg_psnr_ssim")
    @patch("exercise9.Image.new")
    def test_run_battle_calls_ffmpeg_and_compares(
        self, mock_image_new, mock_get_metrics, mock_getsize, mock_image_open, mock_run
    ):
        """run_battle executes FFmpeg commands and compares metrics."""
        # Setup mocks
        mock_run.return_value = MagicMock(returncode=0)
        mock_getsize.side_effect = [5000, 4800]  # JPEG size, JPEG2000 size
        
        # Mock PIL Image operations - need proper size attributes for paste
        mock_orig = MagicMock()
        mock_orig.size = (100, 100)
        mock_orig.convert.return_value = mock_orig
        mock_orig.crop.return_value = mock_orig
        
        mock_jpg = MagicMock()
        mock_jpg.size = (100, 100)
        mock_jpg.convert.return_value = mock_jpg
        mock_jpg.crop.return_value = mock_jpg
        
        mock_j2k = MagicMock()
        mock_j2k.size = (100, 100)
        mock_j2k.convert.return_value = mock_j2k
        mock_j2k.crop.return_value = mock_j2k
        
        # Mock comparison image
        mock_comparison = MagicMock()
        mock_image_new.return_value = mock_comparison
        
        def open_side_effect(path):
            if "battle_ref" in path:
                return mock_orig
            elif "jpeg.jpg" in path:
                return mock_jpg
            elif "jpeg2000.jp2" in path:
                return mock_j2k
            return mock_orig
        
        mock_image_open.side_effect = open_side_effect
        
        # Mock metrics
        mock_get_metrics.side_effect = [
            (30.5, 0.92),  # JPEG PSNR, SSIM
            (32.1, 0.95),  # JPEG2000 PSNR, SSIM
        ]
        
        # Call function
        run_battle("input.jpg", jpeg_q=31, j2k_compression_ratio=24)
        
        # Verify FFmpeg called twice (JPEG + JPEG2000)
        assert mock_run.call_count == 2
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("libopenjpeg" in c for c in calls)
        assert any("-q:v 31" in c for c in calls)
        
        # Verify metrics called
        assert mock_get_metrics.call_count == 2
        
        # Verify comparison image created and saved
        mock_image_new.assert_called_once()
        mock_comparison.paste.assert_called()
        mock_comparison.save.assert_called_once()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])