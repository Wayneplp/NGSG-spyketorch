from __future__ import annotations

import math
import unittest

from src.continual.role_train import RoleTrainCompetitionTracker, winner_distribution_metrics


class RoleTrainCompetitionTests(unittest.TestCase):
    def test_winner_distribution_monopoly(self) -> None:
        counts = [1000] + [0] * 199
        metrics = winner_distribution_metrics(counts)
        self.assertEqual(metrics["unique_winners"], 1.0)
        self.assertAlmostEqual(metrics["top1_winner_share"], 1.0)
        self.assertAlmostEqual(metrics["top5_winner_share"], 1.0)
        self.assertAlmostEqual(metrics["winner_entropy"], 0.0)

    def test_winner_distribution_uniform_ten(self) -> None:
        counts = [100] * 10 + [0] * 190
        metrics = winner_distribution_metrics(counts)
        self.assertEqual(metrics["unique_winners"], 10.0)
        self.assertAlmostEqual(metrics["top1_winner_share"], 0.1)
        self.assertAlmostEqual(metrics["top5_winner_share"], 0.5)
        self.assertAlmostEqual(metrics["winner_entropy"], math.log(10.0), places=6)
        self.assertAlmostEqual(metrics["winner_entropy_normalized"], 1.0, places=6)

    def test_tracker_records_gated_skip(self) -> None:
        tracker = RoleTrainCompetitionTracker(num_neurons=4)
        tracker.begin_epoch()
        tracker.record_sample(
            winner_idx=0,
            forward_silent=False,
            stdp_eligible=True,
            multiplier=0.0,
            update_applied=False,
            role_name="stable",
        )
        tracker.record_sample(
            winner_idx=2,
            forward_silent=False,
            stdp_eligible=True,
            multiplier=1.0,
            update_applied=True,
            role_name="reserve",
        )
        epoch_summary = tracker.end_epoch(0)
        self.assertEqual(epoch_summary["unique_winners"], 2.0)
        self.assertEqual(epoch_summary["gated_skipped_stdp_samples"], 1.0)
        self.assertEqual(epoch_summary["forward_winner_gated_no_update_samples"], 1.0)
        self.assertTrue(epoch_summary["has_forward_winner_gated_no_update"])
        self.assertAlmostEqual(epoch_summary["forward_winner_gated_no_update_rate"], 0.5)
        self.assertEqual(epoch_summary["effective_stdp_updates"], 1.0)
        self.assertAlmostEqual(epoch_summary["reserve_update_share"], 1.0)
        self.assertAlmostEqual(epoch_summary["role_update_share"]["reserve"], 1.0)
        self.assertAlmostEqual(epoch_summary["role_update_share"]["shared"], 0.0)

        stage = tracker.stage_summary()
        self.assertEqual(stage["gated_skipped_stdp_samples"], 1.0)
        self.assertEqual(stage["forward_winner_gated_no_update_samples"], 1.0)
        self.assertTrue(stage["has_forward_winner_gated_no_update"])
        self.assertEqual(len(stage["epoch_history"]), 1)


if __name__ == "__main__":
    unittest.main()
