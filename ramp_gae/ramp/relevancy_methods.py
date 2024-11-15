import torch
from torch import nn
import torch.nn.functional as F
from ramp_gae.utils import normalize_relevance, NormalizationType
from ramp_gae.ramp.models import TimmVisionTransformer
from typing import Optional


class RelevancyMethod:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def relevancy(self, x: torch.tensor, y: Optional[torch.tensor]) -> torch.tensor:
        relevance = self._generate_relevancy(x=x, y=y)
        # make sure to return normalized relevance
        return normalize_relevance(relevance=relevance, normalization_type=NormalizationType.MAX_TO_ONE)

    def _generate_relevancy(self, x: torch.tensor, y: Optional[torch.tensor]) -> torch.tensor:
        raise NotImplementedError("relevancy method needs to be implemented")


class IntRelevancyMethod(RelevancyMethod):
    def _generate_relevancy(self, x: torch.tensor, y: Optional[torch.tensor], rule: str = "intline") -> torch.tensor:
        x = x.to(self.device)
        x.requires_grad = True

        o = self.model(x)
        num_classes = o.shape[-1]

        # take max prediction as y if not supplied
        if not y:
            y = o.max(-1)[1]
        y = y.to(self.device)

        if isinstance(self.model, TimmVisionTransformer):
            positive_output_relevance = F.one_hot(y, num_classes=num_classes).to(self.device)
            negative_output_relevance = F.one_hot(y, num_classes=num_classes).to(self.device)

            positive_relevance = self.model.backward_prel(positive_output_relevance, rule)
            self.model.zero_grad()

            o = self.model(x)
            negative_relevance = self.model.backward_prel(negative_output_relevance, rule)
            relevance = normalize_relevance(
                relevance=positive_relevance, normalization_type=NormalizationType.SUM_TO_ONE
            ) - normalize_relevance(relevance=negative_relevance, normalization_type=NormalizationType.SUM_TO_ONE)
        else:
            output_relevance = 1.001 * F.one_hot(y, num_classes=num_classes).to(self.device) - torch.ones_like(o) / 1000
            relevance = self.model.backward_prel(output_relevance, rule)

        return relevance
