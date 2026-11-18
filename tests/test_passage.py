import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes

from training import fit_weight, weigh_passage


class PassageTests(unittest.TestCase):
    def test_stopped_animal_gap_group_and_end_of_recording(self):
        single = (1, (190, 64))
        empty = (0, None)
        sequence = ([empty] + [single] * 20 + [empty] * 2
                    + [single, empty, single] + [empty] * 2
                    + [single, (2, None), single] + [empty] * 2 + [single])
        rows = list(weigh_passage.separar_passagens(sequence, 2, 2, 2.5))
        self.assertEqual([row["status"] for row in rows], ["ok", "revisar", "revisar", "revisar"])
        self.assertEqual(rows[0]["amostras"], 20)
        self.assertEqual(rows[1]["motivo"], "deteccao_interrompida")
        self.assertEqual(rows[2]["motivo"], "animais_juntos")
        self.assertIn("fim_sem_intervalo_livre", rows[3]["motivo"])
        self.assertIn("poucas_amostras", rows[3]["motivo"])

    def test_cli_marker_only_in_reference_and_weight_import(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            ref = np.full((160, 240), 255, np.uint8)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            ref[40:120, 20:100] = cv2.aruco.generateImageMarker(dictionary, 0, 80)
            photo = folder / "ref.png"
            self.assertTrue(cv2.imwrite(str(photo), ref))
            path = folder / "video.mp4"
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (240, 160))
            self.assertTrue(out.isOpened())
            for occupied in [False, True, True, False, False, True, True]:
                frame = np.full((160, 240, 3), 230, np.uint8)
                if occupied:
                    cv2.ellipse(frame, (120, 80), (60, 18), 0, 0, 360, (30, 30, 30), -1)
                out.write(frame)
            out.release()

            def detect(roi, **options):
                self.assertEqual(roi.shape[:2], (140, 200))
                self.assertEqual(options["classes"], [0])
                occupied = roi[70, 100, 0] < 100
                # Coordinates relative to the cropped ROI.
                boxes = torch.tensor([[36, 47, 164, 93, 0.9, 0]]) if occupied else torch.empty((0, 6))
                return [SimpleNamespace(boxes=Boxes(boxes, orig_shape=roi.shape[:2]), masks=None)]

            model = Mock(task="detect", names={0: "cattle"})
            model.predict.side_effect = detect
            result = folder / "out.csv"
            args = ["weigh_passage", str(path), "--referencia", str(photo), "--marker-cm", "79",
                    "--intervalo-vazio", "2", "--min-amostras", "2", "--area", "20", "10", "220", "150",
                    "--device", "cpu", "-o", str(result)]
            with patch("sys.argv", args), patch.object(weigh_passage, "YOLO", return_value=model):
                weigh_passage.main()
            with result.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["status"] for r in rows], ["ok", "revisar"])
            self.assertEqual(rows[0]["primeiro_frame"], "2")
            self.assertAlmostEqual(float(rows[0]["comprimento_cm"]), 120, delta=5)
            with result.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows([{**r, "peso_kg": 400} for r in rows])
            self.assertEqual(len(fit_weight.carregar(result)), 1)

    def test_rejects_mismatched_resolution(self):
        cap = Mock()
        cap.read.return_value = (True, np.zeros((50, 100, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "resolução"):
            next(weigh_passage.observar(cap, np.zeros((100, 100, 3)), None, None, 1, {}))
