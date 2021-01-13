# pylint: disable=no-member

"""Contains utils used within models package."""

import torchvision.transforms as T
import torch

DENORMALIZER = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[
    1/0.229, 1/0.224, 1/0.225])

NORMALIZER = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
    0.229, 0.224, 0.225])


def clipped_renormalize(normalized_image: torch.Tensor) -> torch.Tensor:
    """
    Takes normalized image and sequentially performs:
    1. Denormalization.
    2. Clipping it to range [0, 1].
    3. Normalization.
    """
    with torch.no_grad():
        return NORMALIZER(torch.clamp(DENORMALIZER(normalized_image), min=0, max=1))
