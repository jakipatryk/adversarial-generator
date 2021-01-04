import torch


class DummyClasifier(torch.nn.Module):
    def __init__(self, predictions: torch.Tensor):
        super(DummyClasifier, self).__init__()
        self.predictions = predictions

    def forward(self, x: torch.Tensor):
        return self.predictions
