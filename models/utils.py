"""Contains utils used within models package."""

import torchvision.transforms as T

DENORMALIZER = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[
    1/0.229, 1/0.224, 1/0.225])

NORMALIZER = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
    0.229, 0.224, 0.225])
