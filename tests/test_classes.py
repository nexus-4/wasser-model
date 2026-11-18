from types import SimpleNamespace
import unittest

from processor import resolve_class_ids
from training.pre_annotate import ANIMAL_LIKE_COCO, resolve_pre_annotate_classes


class ClassResolutionTests(unittest.TestCase):
    def test_bootstrap_and_detection_have_distinct_targets(self):
        names = dict.fromkeys(range(80), "other")
        names.update({14: "bird", 19: "cow"})
        model = SimpleNamespace(task="detect", names=names)
        self.assertEqual(resolve_class_ids(model), [19])
        self.assertEqual(resolve_pre_annotate_classes(model), ANIMAL_LIKE_COCO)
        self.assertEqual(resolve_pre_annotate_classes(model, "19"), [19])
        model.task = "classify"
        with self.assertRaises(ValueError):
            resolve_pre_annotate_classes(model)

    def test_custom_weights(self):
        model = SimpleNamespace(names=["cattle"])
        self.assertEqual(resolve_class_ids(model), [0])
        self.assertEqual(resolve_pre_annotate_classes(model), [0])
        with self.assertRaises(ValueError):
            resolve_class_ids(model, "19")
        model.names = ["cow", "dog", "bezerro"]
        self.assertEqual(resolve_class_ids(model), [0, 2])
