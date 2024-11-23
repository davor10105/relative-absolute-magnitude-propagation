import torch
from torch import nn
import torch.nn.functional as F
from abslrp_gae.utils import normalize_relevance, NormalizationType
from typing import Optional


class RelevancyMethod:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def relevancy(self, x: torch.tensor, y: Optional[torch.tensor] = None) -> torch.tensor:
        relevance = self._generate_relevancy(x=x, y=y)
        # make sure to return normalized relevance
        return normalize_relevance(relevance=relevance.detach().cpu(), normalization_type=NormalizationType.MAX_TO_ONE)

    def _generate_relevancy(self, x: torch.tensor, y: Optional[torch.tensor]) -> torch.tensor:
        raise NotImplementedError("relevancy method needs to be implemented")


class AbsLRPRelevancyMethod(RelevancyMethod):
    def _generate_relevancy(self, x: torch.tensor, y: Optional[torch.tensor] = None) -> torch.tensor:
        x = x.to(self.device)
        x.requires_grad = True

        output = self.model(x)

        # take max prediction as y if not supplied
        if y is None:
            y = output.max(-1)[1]
        y = y.to(self.device)

        # sglrp rule
        # output_relevance = torch.autograd.grad(output.softmax(-1)[torch.arange(output.shape[0]), y].sum(), output)[0]
        # clrp rule to provide contrastive relevance maps
        output_relevance = torch.ones_like(output)
        positive_mask = F.one_hot(y, num_classes=output.shape[-1])

        positive_relevance = self.model.explain(
            output_relevance=normalize_relevance(
                (output_relevance * positive_mask).abs(),
                normalization_type=NormalizationType.SUM_TO_ONE,
            ),
            retain_graph=True,
        )

        negative_relevance = self.model.explain(
            output_relevance=normalize_relevance(
                (output_relevance * (1 - positive_mask)).abs(), normalization_type=NormalizationType.SUM_TO_ONE
            )
        )

        # final relevance is the difference between positive and negative relevance
        relevance = positive_relevance - negative_relevance

        return relevance
