import unittest

from src.analysis.role_path_selector import (
    ROLE_PATH_FEATURE_NAMES,
    apply_role_path_policy,
    extract_role_path_features,
    role_path_label,
)


class RolePathSelectorTests(unittest.TestCase):
    def test_features_are_7d(self):
        scores = [0.1] * 10
        scores[1] = 5.0
        scores[2] = 4.0
        scores[7] = 3.0
        scores[8] = 2.5
        feats = extract_role_path_features(scores, stable_indices=[1, 2], plastic_indices=[7, 8])
        self.assertEqual(len(feats), len(ROLE_PATH_FEATURE_NAMES))
        self.assertEqual(feats[0], 5.0)
        self.assertEqual(feats[3], 3.0)

    def test_label_disagree_only(self):
        self.assertEqual(role_path_label(True, False), 0)
        self.assertEqual(role_path_label(False, True), 1)
        self.assertIsNone(role_path_label(True, True))
        self.assertIsNone(role_path_label(False, False))

    def test_apply_policy(self):
        rows = [
            {
                "true_task": 0,
                "true_class": 1,
                "stable_prediction": 1,
                "plastic_prediction": 2,
                "stable_correct": True,
                "plastic_correct": False,
            },
            {
                "true_task": 1,
                "true_class": 3,
                "stable_prediction": 0,
                "plastic_prediction": 3,
                "stable_correct": False,
                "plastic_correct": True,
            },
        ]
        metrics, annotated = apply_role_path_policy(rows, probabilities=[0.1, 0.9], threshold=0.5)
        self.assertEqual(annotated[0]["selector_prediction"], 1)
        self.assertEqual(annotated[1]["selector_prediction"], 3)
        self.assertAlmostEqual(metrics["macro_average"], 1.0)


if __name__ == "__main__":
    unittest.main()
