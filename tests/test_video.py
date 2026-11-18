import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from processor import prepare_browser_video


class PlaybackTests(unittest.TestCase):
    def test_failed_conversion_keeps_download(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "result.mp4"
            path.write_bytes(b"original")
            with patch("processor.shutil.which", return_value=None):
                self.assertFalse(prepare_browser_video(path))
            with patch("processor.shutil.which", return_value="ffmpeg"), patch(
                "processor.subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
            ):
                self.assertFalse(prepare_browser_video(path))
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(path.parent.iterdir()), [path])

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg with libx264")
    def test_h264_is_readable(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "result.mp4"
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            for _ in range(3):
                out.write(np.zeros((48, 64, 3), np.uint8))
            out.release()
            self.assertTrue(prepare_browser_video(path))
            cap = cv2.VideoCapture(str(path))
            try:
                codec = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").lower()
                self.assertIn(codec, (b"avc1", b"h264"))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                self.assertTrue(cap.read()[0])
            finally:
                cap.release()
