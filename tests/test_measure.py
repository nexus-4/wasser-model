import unittest

import cv2
import numpy as np
import torch

from training.measure_cattle import isolado, medir_deteccao, medida_orientada, toca_borda


class MeasurementTests(unittest.TestCase):
    def test_otsu_for_light_and_dark_coats(self):
        for background, animal in [(25, 210), (210, 25)]:
            roi = np.full((64, 120, 3), background, np.uint8)
            cv2.ellipse(roi, (60, 32), (54, 22), 0, 0, 360, (animal,) * 3, -1)
            length, width, _ = medida_orientada(roi)
            self.assertAlmostEqual(length, 108, delta=4)
            self.assertAlmostEqual(width, 44, delta=3)

    def test_instance_index_and_coordinates(self):
        frame = np.zeros((100, 150, 3), np.uint8)
        masks = torch.zeros((2, 100, 150), dtype=torch.uint8)
        masks[1, 40:61, 50:111] = 1
        box = (42, 30, 120, 70)
        self.assertIsNone(medir_deteccao(frame, box, masks, 0))
        self.assertEqual(medir_deteccao(frame, box, masks, 1)[:2], (60, 20))
        self.assertIsNone(medir_deteccao(frame, box, None, 1))

    def test_isolation_and_border(self):
        box = (20, 20, 40, 40)
        self.assertTrue(isolado(box, [box]))
        self.assertFalse(isolado(box, [box, (41, 22, 62, 40)]))
        self.assertFalse(toca_borda(box, 100, 100))
        self.assertTrue(toca_borda((4, 12, 30, 30), 100, 100))
