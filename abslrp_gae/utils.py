import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
from enum import Enum
from typing import Optional
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from PIL import Image


class NormalizationType(Enum):
    SUM_TO_ONE = "abs"
    MAX_TO_ONE = "max"


@torch.no_grad()
def normalize_relevance(relevance: torch.tensor, normalization_type: NormalizationType) -> torch.tensor:
    shape = relevance.shape
    relevance = relevance.flatten(1)
    # normalize using the chosen mode
    if normalization_type == NormalizationType.SUM_TO_ONE:
        relevance = relevance / (relevance.abs().sum(-1, keepdim=True) + 1e-9)
    else:
        relevance = relevance / (relevance.abs().max(-1, keepdim=True)[0] + 1e-9)
    relevance = relevance.view(*shape)

    return relevance


@torch.no_grad()
def renormalize_images(x: torch.tensor, is_vision_transformer_input: bool = False) -> torch.tensor:
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN)
    std = torch.tensor(IMAGENET_DEFAULT_STD)
    if is_vision_transformer_input:
        mean = torch.tensor(IMAGENET_INCEPTION_MEAN)
        std = torch.tensor(IMAGENET_INCEPTION_STD)
    x = x * std[None, :, None, None] + mean[None, :, None, None]
    return x


def preprocess_image(image: Image, is_vision_transformer_input: bool = False) -> torch.tensor:
    image = pil_to_tensor(image) / 255
    transforms = T.Compose(
        [
            T.Resize(size=(256, 256), antialias=True),
            T.CenterCrop(size=(224, 224)),
            (
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                if not is_vision_transformer_input
                else T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            ),
        ]
    )
    image = transforms(image)
    return image


def visualize_batch(x: torch.tensor, relevance: torch.tensor, is_vision_transformer_input: bool = False) -> None:
    num_images = x.shape[0]
    x, relevance = x.detach().cpu(), normalize_relevance(
        relevance.sum(1).detach().cpu(), normalization_type=NormalizationType.MAX_TO_ONE
    )
    x = renormalize_images(x, is_vision_transformer_input).permute(0, 2, 3, 1)
    fig, axs = plt.subplots(num_images, 2)
    for i, (xx, rr) in enumerate(zip(x, relevance)):
        if num_images > 1:
            axs[i, 0].imshow(xx)
            axs[i, 1].imshow(rr, cmap="seismic", vmin=-1, vmax=1)
            axs[i, 0].axis("off")
            axs[i, 1].axis("off")
        else:
            axs[0].imshow(xx)
            axs[1].imshow(rr, cmap="seismic", vmin=-1, vmax=1)
            axs[0].axis("off")
            axs[1].axis("off")
    plt.show()
