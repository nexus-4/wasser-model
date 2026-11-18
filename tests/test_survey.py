from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ultralytics.engine.results import Boxes

from training import survey_count


class SurveyTests(unittest.TestCase):
    def test_chain_preserves_original_ids_and_only_sums_observations(self):
        fragments = {
            tid: dict(id=tid, primeiro_frame=start, ultimo_frame=start+4, deteccoes=5,
                      x_inicial=start, y_inicial=4, x_final=start+4, y_final=4)
            for tid, start in [(8, 12), (12, 30), (25, 45)]
        }
        audit = []
        ids, tracks = survey_count.consolidar(fragments, {12: 8, 25: 12}, audit)
        self.assertEqual(ids, {8: 8, 12: 8, 25: 8})
        self.assertEqual(tracks[8]["ids_origem"], "8,12,25")
        self.assertEqual(tracks[8]["deteccoes"], 15)
        self.assertEqual((tracks[8]["primeiro_frame"], tracks[8]["ultimo_frame"]), (12, 49))
        self.assertEqual([(r["id_final"], r["id_anterior"], r["id_novo"]) for r in audit],
                         [(8, 8, 12), (8, 12, 25)])

    def test_replay_in_slow_motion_is_separated_from_original(self):
        rng = np.random.default_rng(17)
        # Smooth movement within one shot, followed by a cut back to the beginning.
        background = rng.integers(10, 230, (144, 192, 3), dtype=np.uint8)
        import cv2
        background = cv2.GaussianBlur(background, (15, 15), 4)
        shot = [np.roll(background, i, axis=1) for i in range(50)]
        frames = shot + [f for f in shot for _ in range(2)]
        cap = Mock()
        cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        sections = survey_count.dividir_trechos(cap, 1)
        self.assertEqual([(a, b) for a, b, _ in sections], [(0, 50), (50, 150)])
        original, replay = (s[2] for s in sections)
        self.assertTrue(survey_count.provavel_replay(replay, original))
        self.assertFalse(survey_count.provavel_replay(replay[::-1], original))
        self.assertFalse(survey_count.provavel_replay(np.repeat(original[:1], 10, axis=0), original))

    def test_repeated_detections_count_once_and_transient_is_unconfirmed(self):
        video = Mock()
        video.read.side_effect = [(True, np.zeros((180, 180, 3), np.uint8)) for _ in range(16)]
        stable = [20, 40, 50, 100, .8, 0]
        transient = [100, 40, 130, 100, .9, 0]
        timeline = [[stable, transient], [stable], [stable], [], [stable], [stable], [stable], [stable]]
        boxes = [Boxes(np.array(b, dtype=np.float32).reshape(-1, 6), (180, 180)) for b in timeline]
        writer = Mock()
        with patch.object(survey_count, "pasture_boxes", side_effect=boxes):
            rows = survey_count.rastrear(video, 10, 18, SimpleNamespace(), {}, 5, writer)
        self.assertEqual(writer.write.call_count, 8)
        self.assertEqual(sum(r["confirmada"] for r in rows), 1)
        counted = next(r for r in rows if r["confirmada"])
        self.assertEqual((counted["primeiro_frame"], counted["ultimo_frame"]), (10, 17))
        self.assertEqual(counted["deteccoes"], 7)
