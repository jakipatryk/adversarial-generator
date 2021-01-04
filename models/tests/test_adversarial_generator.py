import unittest
import torch
from .dummy_adversarial_generator import DummyAdversarialGenerator
from .dummy_clasifier import DummyClasifier
from ..model import Model


class TestAdversarialGenerator(unittest.TestCase):
    def test_generate(self):
        dummy_adversarial_generator = DummyAdversarialGenerator(
            Model(
                DummyClasifier(torch.tensor([1., 2., 3.])),
                "desc",
                ["class_1", "class_2", "class_3"]
            ),
            "desc")
        image = torch.ones(3, 256, 256)
        adv_image = dummy_adversarial_generator.generate(image)
        self.assertTrue(torch.allclose(adv_image, image))
