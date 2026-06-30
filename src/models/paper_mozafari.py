from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from SpykeTorch import functional as sf
from SpykeTorch import snn, utils


@dataclass(frozen=True)
class PaperMozafariConfig:
    time_steps: int = 15
    num_classes: int = 10
    neurons_per_class: int = 20
    s1_maps: int = 30
    s2_maps: int = 250
    s3_neurons: int = 200
    conv1_kernel: int = 5
    conv2_kernel: int = 3
    conv3_kernel: int = 5
    conv1_threshold: float = 15.0
    conv2_threshold: float = 10.0
    conv1_kwta: int = 5
    conv2_kwta: int = 8
    conv1_inhibition_radius: int = 0
    conv2_inhibition_radius: int = 0
    conv3_inhibition_radius: int = 0
    weight_mean: float = 0.8
    weight_std: float = 0.05
    filter_threshold: float = 50.0
    local_normalization_radius: int = 8

    def validate(self) -> None:
        if self.s3_neurons != self.num_classes * self.neurons_per_class:
            raise ValueError("s3_neurons must equal num_classes * neurons_per_class.")


class PaperMozafariMNIST2018(nn.Module):
    """Port of dmitryanton68/continuous_learning MozafariMNIST2018."""

    paper_source_compatible = True

    def __init__(self, config: Optional[PaperMozafariConfig] = None) -> None:
        super().__init__()
        self.config = config or PaperMozafariConfig()
        self.config.validate()

        self.conv1 = snn.Convolution(6, self.config.s1_maps, self.config.conv1_kernel, self.config.weight_mean, self.config.weight_std)
        self.conv1_t = self.config.conv1_threshold
        self.k1 = self.config.conv1_kwta
        self.r1 = self.config.conv1_inhibition_radius

        self.conv2 = snn.Convolution(self.config.s1_maps, self.config.s2_maps, self.config.conv2_kernel, self.config.weight_mean, self.config.weight_std)
        self.conv2_t = self.config.conv2_threshold
        self.k2 = self.config.conv2_kwta
        self.r2 = self.config.conv2_inhibition_radius

        self.conv3 = snn.Convolution(self.config.s2_maps, self.config.s3_neurons, self.config.conv3_kernel, self.config.weight_mean, self.config.weight_std)
        self.r3 = self.config.conv3_inhibition_radius

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        self.max_ap = Parameter(torch.Tensor([0.15]), requires_grad=False)

        self.decision_map = []
        for class_idx in range(self.config.num_classes):
            self.decision_map.extend([class_idx] * self.config.neurons_per_class)

        self.ctx: Dict[str, Any] = {"input_spikes": None, "potentials": None, "output_spikes": None, "winners": None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

        kernels = [
            utils.DoGKernel(3, 3 / 9, 6 / 9),
            utils.DoGKernel(3, 6 / 9, 3 / 9),
            utils.DoGKernel(7, 7 / 9, 14 / 9),
            utils.DoGKernel(7, 14 / 9, 7 / 9),
            utils.DoGKernel(13, 13 / 9, 26 / 9),
            utils.DoGKernel(13, 26 / 9, 13 / 9),
        ]
        self.source_filter = utils.Filter(kernels, padding=6, thresholds=self.config.filter_threshold)
        self.temporal_transform = utils.Intensity2Latency(self.config.time_steps)

        # Compatibility aliases for older summaries/tools.
        self.s1 = self.conv1
        self.s2 = self.conv2
        self.s3 = self.conv3

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.source_filter.kernels = self.source_filter.kernels.to(device)
        if isinstance(self.source_filter.thresholds, torch.Tensor):
            self.source_filter.thresholds = self.source_filter.thresholds.to(device)
        self.stdp1.to(device)
        self.stdp2.to(device)
        self.stdp3.to(device)
        self.anti_stdp3.to(device)
        return result

    def encode(self, image: Tensor) -> Tensor:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError("Expected single image tensor with shape CxHxW.")
        image = image.to(next(self.parameters()).device)
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.unsqueeze(0).float()
        filtered = self.source_filter(image)
        normalized = sf.local_normalization(filtered, self.config.local_normalization_radius)
        if not torch.any(normalized > 0):
            _, channels, height, width = normalized.shape
            return torch.zeros(
                (self.config.time_steps, channels, height, width),
                dtype=torch.uint8,
                device=normalized.device,
            )
        temporal_image = self.temporal_transform(normalized)
        return temporal_image.sign().byte().to(next(self.parameters()).device)

    def forward(self, input: Tensor, max_layer: int = 3) -> Any:  # type: ignore[override]
        if input.ndim == 3 or (input.ndim == 4 and input.shape[0] == 1):
            input = self.encode(input.squeeze(0) if input.ndim == 4 else input)
        input = sf.pad(input.float(), (2, 2, 2, 2), 0)

        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap.to(ap.device))
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self._store_context(input, pot, spk, winners)
                return spk, pot

            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap.to(ap.device))
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self._store_context(spk_in, pot, spk, winners)
                return spk, pot

            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2))
            pot = self.conv3(spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, self.r3, spk)
            self._store_context(spk_in, pot, spk, winners)
            return self._decision_from_winners(winners)

        pot = self.conv1(input)
        spk, pot = sf.fire(pot, self.conv1_t, True)
        if max_layer == 1:
            return spk, pot
        pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1)))
        spk, pot = sf.fire(pot, self.conv2_t, True)
        if max_layer == 2:
            return spk, pot
        pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2)))
        spk = sf.fire(pot)
        winners = sf.get_k_winners(pot, 1, self.r3, spk)
        return self._decision_from_winners(winners)

    def _store_context(self, input_spikes: Tensor, potentials: Tensor, output_spikes: Tensor, winners: Any) -> None:
        self.ctx["input_spikes"] = input_spikes
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def _decision_from_winners(self, winners: Any) -> int:
        if len(winners) == 0:
            return -1
        return int(self.decision_map[int(winners[0][0])])

    def stdp(self, layer_idx: int) -> None:
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def reset_learning_rates(self) -> None:
        self.stdp1.update_all_learning_rate(0.004, -0.003)
        self.stdp2.update_all_learning_rate(0.004, -0.003)
        self.stdp3.update_all_learning_rate(0.004, -0.003)
        self.anti_stdp3.update_all_learning_rate(-0.004, 0.0005)
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def update_learning_rates(self, stdp_ap: float, stdp_an: float, anti_stdp_ap: float, anti_stdp_an: float) -> None:
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self) -> None:
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self) -> None:
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def extract_s3_input(self, input: Tensor) -> Tensor:
        """Return the pooled C2 spikes used as conv3/S3 input."""
        if input.ndim == 3 or (input.ndim == 4 and input.shape[0] == 1):
            input = self.encode(input.squeeze(0) if input.ndim == 4 else input)
        input = sf.pad(input.float().to(next(self.parameters()).device), (2, 2, 2, 2), 0)
        pot = self.conv1(input)
        spk, _ = sf.fire(pot, self.conv1_t, True)
        spk_in = sf.pad(sf.pooling(spk, 2, 2), (1, 1, 1, 1))
        pot = self.conv2(spk_in)
        spk, _ = sf.fire(pot, self.conv2_t, True)
        return sf.pad(sf.pooling(spk, 3, 3), (2, 2, 2, 2)).detach().contiguous()

    def forward_from_s3_input(self, s3_input: Tensor) -> int:
        """Run only S3/C3/classification from a cached C2 pooled feature tensor."""
        if s3_input.ndim == 5 and s3_input.shape[0] == 1:
            s3_input = s3_input.squeeze(0)
        s3_input = s3_input.float().to(next(self.parameters()).device)
        pot = self.conv3(s3_input)
        spk = sf.fire(pot)
        winners = sf.get_k_winners(pot, 1, self.r3, spk)
        self._store_context(s3_input, pot, spk, winners)
        return self._decision_from_winners(winners)

    def predict_from_s3_input(self, s3_input: Tensor) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = int(self.forward_from_s3_input(s3_input))
        if was_training:
            self.train()
        return output
    def predict_single(self, image: Tensor) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = int(self.forward(image, 3))
        if was_training:
            self.train()
        return output

    def describe(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        payload["implementation"] = "paper_source_mozafari_mnist_2018"
        payload["source_repo"] = "https://github.com/dmitryanton68/continuous_learning"
        payload["spyketorch_modules"] = ["utils.Filter", "utils.Intensity2Latency", "snn.Convolution", "snn.STDP"]
        return payload


def build_paper_mozafari_network(overrides: Optional[Dict[str, Any]] = None) -> PaperMozafariMNIST2018:
    allowed = set(PaperMozafariConfig.__dataclass_fields__)
    filtered = {key: value for key, value in (overrides or {}).items() if key in allowed}
    return PaperMozafariMNIST2018(PaperMozafariConfig(**filtered))
