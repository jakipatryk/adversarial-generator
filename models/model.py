# pylint: disable=no-member

"""Contains class that is used to create instances of models exported for the Django app."""

from typing import List, Tuple, Callable, Type
import torch
import torch.nn.functional as F
from torchvision import transforms as T


class Model:
    """
    Container for PyTorch neural networks.

    Parameters:
    - classifier (Type[torch.nn.Module]): an instance of a class deriving from torch.nn.Module
    - description (str): a text description of the classifier
    - classes (List[str]): a list of classes that the classifier classifies
    - preprocessing_function (torch.Tensor -> torch.Tensor) [optional]:
        should NOT normalize an image
    """

    def __init__(
            self,
            classifier: Type[torch.nn.Module],
            description: str,
            classes: List[str],
            preprocessing_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
        self.classifier = classifier
        self.description = description
        self.classes = classes
        self.preprocessing_function = preprocessing_function

    def predict(self, raw_image: torch.Tensor, normalized=False) -> Tuple[str, float]:
        """
        Classifies given image.

        Arguments:
        - raw_image (torch.Tensor): image with shape [channels, height, width] to classify
        - normalized (bool) [optional]: is raw_image normalized,
            if not simple normalization will be performed

        Returns:
        Tuple[str, float]: 1st element is predicted class, 2nd element is prediction confidence
        """
        image = self.preprocessing_function(raw_image)
        if not normalized:
            normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
            image = normalizer(image)
        self.classifier.eval()
        prediction = self.classifier(image).data.view(-1)
        probabilities = F.softmax(prediction, dim=0)
        predicted_index = torch.argmax(prediction)
        return self.classes[predicted_index], probabilities[predicted_index].item()

    def __str__(self):
        return f"{type(self.classifier).__name__}: {self.description}"
