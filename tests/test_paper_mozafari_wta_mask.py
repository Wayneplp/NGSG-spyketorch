from __future__ import annotations

import unittest

try:
    import torch
    from SpykeTorch import functional as sf

    from src.models.paper_mozafari import PaperMozafariMNIST2018

    HAS_WTA_DEPS = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    sf = None  # type: ignore[assignment]
    PaperMozafariMNIST2018 = None  # type: ignore[assignment]
    HAS_WTA_DEPS = False


def _winner_tuple(pot, spk=None):
    if spk is None:
        spk = sf.fire(pot)
    winners = sf.get_k_winners(pot, 1, 0, spk)
    if len(winners) == 0:
        raise AssertionError("Expected at least one S3 WTA winner.")
    return tuple(int(value) for value in winners[0])


def _mask_model(allow_mask):
    model = PaperMozafariMNIST2018.__new__(PaperMozafariMNIST2018)
    object.__setattr__(model, "s3_wta_allow_mask", allow_mask)
    return model


@unittest.skipUnless(HAS_WTA_DEPS, "torch and SpykeTorch are required for WTA mask tests")
class PaperMozafariS3WTAMaskTests(unittest.TestCase):
    def test_all_open_mask_matches_raw_official_wta(self) -> None:
        raw_pot = torch.zeros((3, 4, 2, 2), dtype=torch.float32)
        raw_pot[0, 2, 0, 1] = 5.0
        raw_pot[1, 1, 1, 0] = 9.0
        raw_pot[2, 3, 0, 0] = 10.0
        model = _mask_model(torch.ones(4, dtype=torch.bool))

        masked_pot, masked_spk = model._s3_wta_inputs(raw_pot)

        self.assertTrue(torch.equal(masked_pot, raw_pot))
        self.assertTrue(torch.equal(masked_spk, sf.fire(raw_pot)))
        self.assertEqual(_winner_tuple(masked_pot, masked_spk), _winner_tuple(raw_pot))

    def test_zero_mask_keeps_potential_and_spike_legal(self) -> None:
        raw_pot = torch.ones((3, 4, 2, 2), dtype=torch.float32)
        raw_pot[:, 0] = 7.0
        allow_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
        model = _mask_model(allow_mask)

        masked_pot, masked_spk = model._s3_wta_inputs(raw_pot)

        self.assertTrue(torch.isfinite(masked_pot).all())
        self.assertTrue(torch.all((masked_spk == 0) | (masked_spk == 1)))
        self.assertTrue(torch.equal(masked_pot[:, ~allow_mask], torch.zeros_like(masked_pot[:, ~allow_mask])))
        self.assertTrue(torch.equal(masked_spk[:, ~allow_mask], torch.zeros_like(masked_spk[:, ~allow_mask])))

    def test_allowed_all_200_winner_is_unchanged(self) -> None:
        raw_pot = torch.zeros((3, 4, 2, 2), dtype=torch.float32)
        raw_pot[0, 2, 0, 1] = 5.0
        raw_pot[1, 1, 1, 0] = 9.0
        raw_pot[2, 2, 0, 0] = 10.0
        raw_pot[2, 3, 1, 1] = 4.0
        allow_mask = torch.tensor([False, True, True, False], dtype=torch.bool)
        model = _mask_model(allow_mask)

        winner_all = _winner_tuple(raw_pot)
        masked_pot, masked_spk = model._s3_wta_inputs(raw_pot)
        winner_masked = _winner_tuple(masked_pot, masked_spk)

        self.assertTrue(bool(allow_mask[winner_all[0]].item()))
        self.assertEqual(winner_masked, winner_all)

    def test_zero_mask_matches_feature_slicing_reference(self) -> None:
        raw_pot = torch.zeros((4, 5, 2, 2), dtype=torch.float32)
        raw_pot[0, 0, 0, 0] = 3.0
        raw_pot[1, 2, 0, 1] = 8.0
        raw_pot[3, 3, 1, 0] = 9.0
        raw_pot[3, 4, 1, 1] = 10.0
        allow_mask = torch.tensor([False, False, True, True, False], dtype=torch.bool)
        model = _mask_model(allow_mask)

        masked_pot, masked_spk = model._s3_wta_inputs(raw_pot)
        masked_winner = _winner_tuple(masked_pot, masked_spk)

        allowed_idx = torch.where(allow_mask)[0]
        subset_pot = raw_pot[:, allowed_idx, :, :]
        local_winner = _winner_tuple(subset_pot)
        mapped_winner = (
            int(allowed_idx[local_winner[0]].item()),
            local_winner[1],
            local_winner[2],
        )

        self.assertEqual(masked_winner, mapped_winner)
        self.assertTrue(bool(allow_mask[masked_winner[0]].item()))

    def test_mask_also_supports_feature_first_3d_potentials(self) -> None:
        raw_pot = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
        allow_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
        model = _mask_model(allow_mask)

        masked_pot = model._apply_s3_wta_mask(raw_pot)

        self.assertTrue(torch.equal(masked_pot[~allow_mask], torch.zeros_like(masked_pot[~allow_mask])))
        self.assertTrue(torch.equal(masked_pot[allow_mask], raw_pot[allow_mask]))

    def test_invalid_allow_mask_length_raises_even_when_all_open(self) -> None:
        raw_pot = torch.zeros((3, 4, 2, 2), dtype=torch.float32)
        model = _mask_model(torch.ones(5, dtype=torch.bool))

        with self.assertRaises(ValueError):
            model._apply_s3_wta_mask(raw_pot)


if __name__ == "__main__":
    unittest.main()
