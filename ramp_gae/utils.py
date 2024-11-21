import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
from enum import Enum
from typing import Optional


class NormalizationType(Enum):
    SUM_TO_ONE = "abs"
    MAX_TO_ONE = "max"


@torch.no_grad()
def normalize_relevance(
    relevance: torch.tensor, normalization_type: NormalizationType, output_factor: Optional[torch.tensor] = None
) -> torch.tensor:
    shape = relevance.shape
    relevance = relevance.flatten(1)
    # normalize using the chosen mode
    if normalization_type == NormalizationType.SUM_TO_ONE:
        relevance = relevance / (relevance.abs().sum(-1, keepdim=True) + 1e-9)
    else:
        relevance = relevance / (relevance.abs().max(-1, keepdim=True)[0] + 1e-9)
    # scale relevance
    if output_factor is not None:
        relevance = relevance * output_factor
    relevance = relevance.view(*shape)

    return relevance


@torch.no_grad()
def normalize_abs_sum_to_one(x):
    original_shape = x.shape
    x = x.flatten(1)
    x /= x.abs().sum(-1, keepdim=True) + 1e-9
    x = x.view(*original_shape).detach()

    return x


@torch.no_grad()
def normalize_prel(prel):
    prel = prel.detach().clone()
    prel_shape = prel.shape
    prel = prel.flatten(1)
    prel = prel / (prel.abs().sum(-1, keepdim=True) + 1e-9)
    prel = prel / (prel.abs().max(-1, keepdim=True)[0] + 1e-9)
    prel = prel.view(prel_shape)

    return prel


def get_sign(h):
    h_sign = torch.where(h >= 0, 1, -1)
    return h_sign


@torch.no_grad()
def l1_distance(x, y):
    return (x.flatten(1) - y.flatten(1)).abs().sum(-1)


@torch.no_grad()
def l2_distance(x, y):
    return (x.flatten(1) - y.flatten(1)).pow(2).sum(-1).pow(0.5)


@torch.no_grad()
def scale_relevance_with_output(r, o, chosen_index):
    r_shape = r.shape
    r = (
        o[torch.arange(o.shape[0], device=o.device).unsqueeze(0), chosen_index][0].unsqueeze(-1) * r.flatten(1)
    ).detach()
    r = r.view(r_shape)

    return r


@torch.no_grad()
def norm_image(x, is_vit=False):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    if is_vit:
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
    x = ((x * std + mean) * 255).int()

    return x


def preprocess_pil_image(image, is_vit=False):
    image = pil_to_tensor(image) / 255
    transforms = T.Compose(
        [
            T.Resize(size=(256, 256), antialias=True),
            T.CenterCrop(size=(224, 224)),
            (
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if not is_vit
                else T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ),
        ]
    )
    image = transforms(image)
    return image


def visualize_tensor_relevance_batch(x, r, is_vit=False):
    num_images = x.shape[0]
    x, r = x.detach().cpu(), normalize_prel(r.detach().cpu()).sum(1)
    fig, axs = plt.subplots(num_images, 2)
    for i, (xx, rr) in enumerate(zip(x, r)):
        xx = norm_image(xx.permute(1, 2, 0), is_vit=is_vit)
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
