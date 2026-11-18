from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from training import line_count


class LineCountTests(unittest.TestCase):
    def test_cli_counts_crossings_once_through_deadzone(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        centers = [30, 49, 51, 48, 75, 52, 49, 30]
        capture = MagicMock()
        capture.get.return_value = 100
        capture.read.side_effect = [(True, frame)] * len(centers) + [(False, None)]
        model = MagicMock()
        model.names = {0: "cattle"}
        model.track.side_effect = [[SimpleNamespace(boxes=SimpleNamespace(
            id=torch.tensor([1]), xyxy=torch.tensor([[x-2, 48, x+2, 52]])
        ))] for x in centers]
        output = StringIO()
        with (
            patch.object(line_count, "YOLO", return_value=model),
            patch.object(line_count.cv2, "VideoCapture", return_value=capture),
            patch("sys.argv", ["line_count", "sample.mp4", "--device", "cpu"]),
            redirect_stdout(output),
        ):
            line_count.main()
        self.assertIn("Entradas : 1", output.getvalue())
        self.assertIn("Saidas   : 1", output.getvalue())
        self.assertTrue(all(call.kwargs["classes"] == [0] for call in model.track.call_args_list))
