# pylint: disable=no-member,invalid-name,missing-docstring,not-callable

import unittest
from unittest.mock import patch
import torch
from .dummy_clasifier import DummyClasifier
from ..deep_fool import DeepFool
from ..model import Model


class TestDeepFool(unittest.TestCase):
    def test_generate_change_tensor(self):
        with patch('torch.autograd.functional.jacobian') as jacobian:
            jacobian.return_value = torch.ones(3, 3, 256, 256)
            jacobian.return_value[2] = torch.zeros(3, 256, 256)
            deep_fool = DeepFool(
                Model(
                    DummyClasifier(torch.tensor([[1., 2., 3.]])),
                    "desc",
                    ["class_1", "class_2", "class_3"]
                )
            )
            image = torch.ones(3, 256, 256)
            adv_image = deep_fool.generate_change_tensor(image)
            self.assertTrue(torch.allclose(
                adv_image, torch.zeros_like(adv_image), atol=1e-04))
