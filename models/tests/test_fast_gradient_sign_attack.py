# pylint: disable=no-member,invalid-name,missing-docstring,not-callable,protected-access

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
        return torch.tensor([[0., 0., 1.]])


class TestFastGradientSignAttack(unittest.TestCase):
    def setUp(self):
        self.ec_1 = EpsilonClassifier(0.0000001)
        self.ec_2 = EpsilonClassifier(0.6)
        self.ec_3 = EpsilonClassifier(10.)
        self.image = torch.zeros(3, 5, 5)
        self.model_1 = Model(self.ec_1, "desc", [
            "class_1", "class_2", "class_3"])
        self.model_2 = Model(self.ec_2, "desc", [
            "class_1", "class_2", "class_3"])
        self.model_3 = Model(self.ec_3, "desc", [
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
        fgsa_1 = FastGradientSignAttack(self.model_1)
        epsilon_1 = fgsa_1._find_epsilon(
            self.image, torch.ones_like(self.image), 0)
        self.assertEqual(epsilon_1, 0.00875)
        fgsa_2 = FastGradientSignAttack(self.model_2)
        epsilon_2 = fgsa_2._find_epsilon(
            self.image, torch.ones_like(self.image), 0)
        self.assertEqual(epsilon_2, 0.60375)
        fgsa_3 = FastGradientSignAttack(self.model_3)
        epsilon_3 = fgsa_3._find_epsilon(torch.full_like(
            self.image, 20.), torch.ones_like(self.image), 0)
        self.assertEqual(epsilon_3, 2.24)
