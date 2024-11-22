import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from enum import Enum
import matplotlib.pyplot as plt
from typing import Optional
from abslrp_gae.abslrp.relevancy_methods import RelevancyMethod
from abslrp_gae.utils import normalize_relevance, NormalizationType


class MaskOrder(Enum):
    MORF = "morf"
    LERF = "lerf"


class GlobalEvaluationMetric:
    def __init__(self, num_masking_steps: int = 10, device: str = "cuda"):
        self.num_masking_steps = num_masking_steps
        self.device = device
        self.results = {}

    @torch.no_grad()
    def _dice_loss(self, t1: torch.tensor, t2: torch.tensor) -> torch.tensor:
        t1, t2 = t1.flatten(1), t2.flatten(1)
        return (t1 - t2).abs().sum(-1) / (t1.abs().sum(-1) + t2.abs().sum(-1) + 1e-9)

    @torch.no_grad()
    def _calculate_mask_overlap(self, relevance: torch.tensor, mask: torch.tensor) -> torch.tensor:
        return (mask * relevance).flatten(1).sum(-1) / (relevance.flatten(1).sum(-1) + 1e-9)

    @torch.no_grad()
    def _create_mosaic(self, x_batch_small: torch.tensor) -> torch.tensor:
        num_mosaics = x_batch_small.shape[0] // 4
        top_left, top_right, bottom_left, bottom_right = torch.split(x_batch_small, num_mosaics)
        top_row = torch.cat([top_left, top_right], dim=3)
        bottom_row = torch.cat([bottom_left, bottom_right], dim=3)
        x_mosaic = torch.cat([top_row, bottom_row], dim=2)
        return x_mosaic

    @torch.no_grad()
    def _mask_by_relevance(
        self, x_batch: torch.tensor, relevance: torch.tensor, percentage: float, mask_order: MaskOrder
    ) -> torch.tensor:
        x_batch_shape = x_batch.shape
        x_batch = x_batch.flatten(1)
        relevance = relevance.flatten(1)

        # find largest (or smallest) points of relevance
        masked_points = relevance.topk(
            k=int(percentage * relevance.size(1)), dim=1, largest=mask_order == MaskOrder.MORF
        )

        # mask the input with zeroes
        masked_x_batch = x_batch.clone()
        masked_x_batch.scatter_(
            dim=1,
            index=masked_points.indices,
            src=torch.zeros_like(masked_x_batch),
        )
        masked_x_batch = masked_x_batch.reshape(*x_batch_shape)

        return masked_x_batch

    def _stepwise_masking(
        self, x_batch: torch.tensor, y_batch: torch.tensor, model: nn.Module, mask_order: MaskOrder, steps: int = 10
    ) -> tuple[list[torch.tensor], list[torch.tensor], list[torch.tensor]]:
        # prepare lists for saving inputs, outputs and relevances
        save_x, save_o, save_r = [], [], []

        step_size = 1 / steps
        batch_size = x_batch.shape[0]
        masked_x_batch = x_batch.detach().clone()
        for percentage in torch.linspace(step_size, 1, steps):
            model.zero_grad()
            masked_x_batch.requires_grad = True

            # define the loss as a minimization of the total activation of the positive class (aiming for 0 activation)
            o = model(masked_x_batch)
            positive_activations = o[torch.arange(batch_size), y_batch]
            loss = positive_activations.abs().mean()
            loss.backward()

            # obtain input x gradient map
            inputxgrad_activation = (masked_x_batch.grad.data * masked_x_batch.data).abs()

            # save inputs, outputs and gradient activations
            save_x.append(masked_x_batch.detach().cpu())
            save_o.append(positive_activations.detach().cpu())
            save_r.append(inputxgrad_activation.detach().cpu())

            # mask most (or least relevant) input pixels, skip if last step
            if percentage < 1.0:
                masked_x_batch = self._mask_by_relevance(
                    x_batch=masked_x_batch,
                    relevance=inputxgrad_activation,
                    percentage=percentage,
                    mask_order=mask_order,
                )

        return save_x, save_o, save_r

    def run(
        self, relevancy_methods: dict[str, RelevancyMethod], model: nn.Module, dataset: Dataset, batch_size: int
    ) -> dict[str, dict[str, torch.tensor]]:
        assert batch_size % 4 == 0, "batch_size must be divisible by 4"

        num_mosaics = batch_size // 4
        gae_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        results = {
            method_name: {"local_consistency": [], "contrastiveness": [], "gae": []}
            for method_name in relevancy_methods
        }
        for x_batch, y_batch in gae_loader:
            x_batch = x_batch.to(self.device)
            batch_size, _, height, width = x_batch.shape

            # infer initial predictions
            with torch.no_grad():
                o_batch = model(x_batch)

            # choose positive examples and predictions from batch
            mosaic_positive_indices = torch.randint(high=4, size=(num_mosaics,))
            positive_indices = mosaic_positive_indices + torch.arange(num_mosaics) * 4
            positive_x = x_batch[positive_indices]
            positive_y = o_batch.max(-1)[1][positive_indices]

            # calculate morf and lerf objects
            morf_x, morf_o, morf_r = self._stepwise_masking(
                x_batch=positive_x,
                y_batch=positive_y,
                model=model,
                mask_order=MaskOrder.MORF,
                steps=self.num_masking_steps,
            )
            lerf_x, lerf_o, lerf_r = self._stepwise_masking(
                x_batch=positive_x,
                y_batch=positive_y,
                model=model,
                mask_order=MaskOrder.LERF,
                steps=self.num_masking_steps,
            )

            # calculate combined impact map
            summed_morf_r = torch.stack(morf_r, dim=0).sum(0)
            summed_lerf_r = torch.stack(lerf_r, dim=0).sum(0)

            combined_impact_map = normalize_relevance(
                relevance=normalize_relevance(relevance=summed_lerf_r, normalization_type=NormalizationType.SUM_TO_ONE)
                - normalize_relevance(relevance=summed_morf_r, normalization_type=NormalizationType.SUM_TO_ONE),
                normalization_type=NormalizationType.MAX_TO_ONE,
            ).sign()

            # combine to mosaic
            x_batch_small = F.interpolate(x_batch, (height // 2, width // 2))
            x_mosaic = self._create_mosaic(x_batch_small)

            # infer mosaic predictions
            with torch.no_grad():
                o_mosaic = model(x_mosaic)

            # generate score map
            score_batch_small = torch.ones_like(x_batch_small)
            softmax_positive_batch = o_batch[positive_indices].softmax(-1)
            mosaic_ys = y_batch.view(num_mosaics, -1)
            softmax_positive_batch[torch.arange(num_mosaics)[:, None], mosaic_ys]
            mosaic_softmax_scores = softmax_positive_batch[torch.arange(num_mosaics)[:, None], mosaic_ys]
            normalized_mosaic_softmax_scores = (
                2
                * mosaic_softmax_scores
                / mosaic_softmax_scores[torch.arange(num_mosaics), mosaic_positive_indices][:, None]
                - 1
            )
            score_batch_small = score_batch_small * normalized_mosaic_softmax_scores.flatten()[:, None, None, None]
            score_mosaic = self._create_mosaic(score_batch_small).detach().cpu()

            # calculate GAE for each method
            for method_name, relevancy_method in relevancy_methods.items():
                # get positive relevance maps
                relevance = relevancy_method.relevancy(x=positive_x, y=positive_y).relu().detach().cpu()
                # get mosaic relevance map
                mosaic_relevance = relevancy_method.relevancy(x=x_mosaic, y=positive_y).relu().detach().cpu()

                # calculate step-wise differences
                relevance_diffs, output_diffs = [], []
                initial_output = morf_o[0]
                for step_morf_x, step_lerf_x, step_morf_o, step_lerf_o in zip(
                    morf_x[1:], lerf_x[1:], morf_o[1:], lerf_o[1:]
                ):
                    step_morf_r, step_lerf_r = (
                        relevancy_method.relevancy(x=step_morf_x, y=positive_y).relu(),
                        relevancy_method.relevancy(x=step_lerf_x, y=positive_y).relu(),
                    )
                    relevance_diff = self._dice_loss(relevance, step_lerf_r) - self._dice_loss(relevance, step_morf_r)
                    output_diff = step_lerf_o / (initial_output + 1e-9) - step_morf_o / (initial_output + 1e-9)
                    relevance_diffs.append(relevance_diff)
                    output_diffs.append(output_diff)
                relevance_diffs, output_diffs = torch.stack(relevance_diffs, 1), torch.stack(output_diffs, 1)

                # calculate faithfulness
                faithfulness = self._calculate_mask_overlap(relevance=relevance, mask=combined_impact_map)

                # calculate robustness
                robustness = 1 - 2 * self._dice_loss(relevance_diffs, output_diffs)

                # calculate local consistency
                local_consistency = (faithfulness + robustness).relu() / 2

                # calculate contrastiveness
                contrastiveness = self._calculate_mask_overlap(relevance=mosaic_relevance, mask=score_mosaic).relu()

                # calculate GAE
                gae_score = local_consistency * contrastiveness

                # save results
                results[method_name]["local_consistency"].append(local_consistency)
                results[method_name]["contrastiveness"].append(contrastiveness)
                results[method_name]["gae"].append(gae_score)

        results = {
            method_name: {
                "local_consistency": torch.cat(values["local_consistency"]),
                "contrastiveness": torch.cat(values["contrastiveness"]),
                "gae": torch.cat(values["gae"]),
            }
            for method_name, values in results.items()
        }
        self.results = results
        return results

    def plot(self, results: Optional[dict[str, dict[str, torch.tensor]]] = None) -> None:
        # plot last results if results is not passed
        if not results:
            results = self.results
        # calculate means
        properties = ["local_consistency", "contrastiveness", "gae"]
        result_means = {
            method_name: {property: results[method_name][property].mean() for property in properties}
            for method_name in results
        }

        # plot bars
        x = torch.arange(len(results))
        bar_width = 0.2

        _, ax = plt.subplots(figsize=(len(results) * 3, 5))

        for i, property in enumerate(properties):
            property_means = [result_means[method_name][property] for method_name in result_means]
            ax.bar(x + i * bar_width, property_means, width=bar_width, label=property)

        # set title and show
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(results.keys())
        ax.set_xlabel("Properties")
        ax.set_ylabel("Mean Values")
        ax.set_title("Global Attribution Evaluation")
        ax.legend()

        plt.tight_layout()
        plt.show()
