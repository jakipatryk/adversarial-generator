# pylint: disable=no-member

"""Contains base abstract class for all adversarial generators."""

import torch
from torchvision import transforms as T
from .model import Model


class AdversarialGenerator():
    """
    A base class for all adversarial generators.

    Parameters:
    - model (Model): a model that will be used to generate adversarial images for
    - description (str): a text description of the generator
    """

    def __init__(self, model: Model, description: str):
        self.model = model
        self.description = description

    def generate(self, raw_image: torch.Tensor, normalized=False) -> torch.Tensor:
        """
        Generates an adversarial image given some image.

        Arguments:
        - raw_image (torch.Tensor): image with shape [channels, height, width]
        - normalized (bool) [optional]: is raw_image normalized,
            if not simple normalization will be performed

        Returns:
        torch.Tensor: generated adversarial image with the same shape as raw_image
        """
        if not normalized:
            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
            raw_image = normalizer(raw_image)
        preprocessed_image = self.model.preprocessing_function(raw_image)
        change_tensor = self.generate_change_tensor(preprocessed_image)
        denormalizer = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[
            1/0.229, 1/0.224, 1/0.225])
        adversatial_image = denormalizer(preprocessed_image + change_tensor)
        adversatial_image = torch.clamp(adversatial_image, min=0, max=1)
        return adversatial_image

    def generate_change_tensor(self, preprocessed_image: torch.Tensor) -> torch.Tensor:
        """
        Generates a tensor that is supposed to be added to the image to change model prediction.

        It has to be overridden by derived classes.

        Arguments:
        - preprocessed_image (torch.Tensor): normalized and preprocessed
            image with shape [channels, height, width]

        Returns:
        torch.Tensor: tensor to be added to the image to change prediction
        """
        raise NotImplementedError
