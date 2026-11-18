"""Regressões para reassociação no inventário visual."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from processor import DEFAULT_VIDEO_PATH
from training import track_recovery
from training.survey_count import REPO


class TrackRecoveryTests(unittest.TestCase):
    def test_real_gap_and_two_nearby_cattle(self):
        # Última caixa do animal claro no frame 204 e três animais no frame 275.
        video = cv2.VideoCapture(str(REPO / DEFAULT_VIDEO_PATH))
        claro_antes = np.array([354.4, 486.7, 380.4, 549.7])
        try:
            for nome, caixa, esperado in [
                ("claro", [319.8, 538.9, 350.2, 608.5], True),
                ("vizinho à direita", [351.3, 519.2, 387.7, 603.3], False),
                ("vizinho abaixo", [376.9, 597.3, 412.5, 675.7], False),
            ]:
                with self.subTest(nome=nome):
                    self.assertEqual(track_recovery.mesma_aparencia(
                        video, 204, 275, claro_antes, np.array(caixa), 4), esperado)
        finally:
            video.release()

    def test_rejects_overlaps_long_gaps_and_ambiguous_matches(self):
        caixa = np.array([20, 20, 45, 70])
        def links(spans):
            regs = {i: dict(id=i, primeiro_frame=a, ultimo_frame=b, deteccoes=5)
                    for i, (a, b) in enumerate(spans)}
            return track_recovery.recuperar_ids(None, regs, dict.fromkeys(regs, (caixa, caixa)), 5, 30)

        with patch.object(track_recovery, "mesma_aparencia", return_value=True):
            self.assertEqual(links([(0, 10), (10, 20)]), {})
            self.assertEqual(links([(0, 10), (101, 110)]), {})
            self.assertEqual(links([(0, 10), (12, 30), (13, 29)]), {})
            self.assertEqual(links([(0, 10), (0, 11), (14, 29)]), {})
            self.assertEqual(links([(0, 10), (14, 29)]), {1: 0})

    def test_textureless_patch_is_insufficient(self):
        frame = np.full((100, 100), 140, dtype=np.uint8)
        self.assertIsNone(track_recovery.seguir_recorte(
            {0: frame, 1: frame}, [0, 1], np.array([20, 20, 40, 60])))
