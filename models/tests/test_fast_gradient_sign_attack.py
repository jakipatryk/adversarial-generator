import unittest
from unittest.mock import patch
import torch
from ..fast_gradient_sign_attack import FastGradientSignAttack
from ..model import Model


class EpsilonClassifier(torch.nn.Module):
    def __init__(self, epsilon: torch.Tensor):
        super(EpsilonClassifier, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        if torch.sum(torch.abs(x) > self.epsilon) == 0:
            return torch.tensor([[1., 0., 0.]])
        else:
            return torch.tensor([[0., 0., 1.]])


class TestFastGradientSignAttack(unittest.TestCase):
    def setUp(self):
        self.ec_1 = EpsilonClassifier(0.0000001)
        self.ec_2 = EpsilonClassifier(0.6)
        self.image = torch.zeros(3, 5, 5)
        self.model_1 = Model(self.ec_1, "desc", [
                             "class_1", "class_2", "class_3"])
        self.model_2 = Model(self.ec_2, "desc", [
                             "class_1", "class_2", "class_3"])

    def test_generate_change_tensor(self):
        with patch('torch.autograd.grad') as grad:
            grad.return_value = [torch.ones_like(self.image)]
            fgsa_1 = FastGradientSignAttack(self.model_1)
            change_tensor_1 = fgsa_1.generate_change_tensor(self.image)
            pred_1 = self.model_1.predict(self.image + change_tensor_1)
            self.assertEqual(pred_1[0], "class_3")
            fgsa_2 = FastGradientSignAttack(self.model_2)
            change_tensor_2 = fgsa_2.generate_change_tensor(self.image)
            pred_2 = self.model_1.predict(self.image + change_tensor_2)
            self.assertEqual(pred_2[0], "class_3")

    def test__find_epsilon(self):
        ec = EpsilonClassifier(0.0000001)
        model = Model(ec, "desc", ["class_1", "class_2", "class_3"])
        fgsa = FastGradientSignAttack(model)
        epsilon = fgsa._find_epsilon(
            torch.zeros(3, 5, 5), torch.ones(3, 5, 5), 0)
        self.assertEqual(epsilon, 0.00390625)
        ec.epsilon = 0.6
        epsilon = fgsa._find_epsilon(
            torch.zeros(3, 5, 5), torch.ones(3, 5, 5), 0)
        self.assertEqual(epsilon, 0.6015625)
