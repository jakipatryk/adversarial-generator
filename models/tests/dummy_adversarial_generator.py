import torch
from ..adversarial_generator import AdversarialGenerator
from ..model import Model


class DummyAdversarialGenerator(AdversarialGenerator):
    def __init__(self, model: Model, description: str):
        super(DummyAdversarialGenerator, self).__init__(model, description)

    def generate_change_tensor(self, preprocessed_image: torch.Tensor):
        return torch.zeros_like(preprocessed_image)
