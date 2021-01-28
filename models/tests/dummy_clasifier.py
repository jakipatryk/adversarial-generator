# pylint: disable=no-member,invalid-name,missing-docstring

import torch


class DummyClasifier(torch.nn.Module):
    def __init__(self, predictions: torch.Tensor):
        super(DummyClasifier, self).__init__()
        self.predictions = predictions

    def forward(self, _: torch.Tensor):
        return self.predictions
