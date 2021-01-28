# pylint: disable=no-member,invalid-name,missing-docstring,not-callable

import unittest
import torch
from ..model import Model
from .dummy_clasifier import DummyClasifier


class TestModel(unittest.TestCase):

    def test_predict(self):
        dc = DummyClasifier(torch.tensor([1., 2., 3.]))
        model = Model(dc, "desc", ["class_1", "class_2", "class_3"])
        predicted_class, predicted_probability = model.predict(
            torch.ones(3, 256, 256))
        self.assertEqual(predicted_class, "class_3")
        self.assertAlmostEqual(predicted_probability, 0.665240955)

    def test_str(self):
        dc = DummyClasifier(torch.tensor([1., 2., 3.]))
        model = Model(dc, "desc", ["class_1", "class_2", "class_3"])
        self.assertEqual(str(model), "DummyClasifier: desc")
