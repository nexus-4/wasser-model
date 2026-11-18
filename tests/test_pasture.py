from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes

from processor import process_video
from training.eval_pasture_count import erros


class PastureTests(unittest.TestCase):
    def test_counts_without_tracking_and_suppresses_overlaps(self):
        model = Mock(task="detect", names={0: "cattle"})
        raw = torch.tensor([[4, 5, 25, 40, .82, 0], [5, 6, 26, 41, .40, 0], [28, 5, 50, 39, .17, 0]])
        model.predict.side_effect = [
            [SimpleNamespace(boxes=Boxes(boxes, (64, 64)))]
            for boxes in (raw, raw, torch.empty((0, 6)))
        ]
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "in.mp4"
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
            for _ in range(3):
                out.write(np.zeros((64, 64, 3), np.uint8))
            out.release()
            counts = []
            result = process_video(path, folder / "out.mp4", model=model, model_path="sample.pt",
                                   tracker_path=None, counter_mode="pasto", device="cpu", preview_interval=1,
                                   preview_callback=lambda frame, count, heatmap: counts.append(count))
            self.assertEqual(counts, [2, 2, 0])
            self.assertIsNone(result["total_unique"])
            self.assertEqual(result["last_frame_count"], 0)
            model.track.assert_not_called()
            self.assertEqual(model.predict.call_args.kwargs["conf"], 0.10)
            self.assertEqual(model.predict.call_args.kwargs["imgsz"], 416)

    def test_benchmark_pairs_each_center_and_box_only_once(self):
        boxes = [(0, 0, 20, 10), (0, 0, 10, 10), (0, 0, 10, 10), (75, 40, 80, 45)]
        self.assertEqual(erros(boxes, [(5, 5), (15, 5), (40, 40)]), (1, 1, 1))
