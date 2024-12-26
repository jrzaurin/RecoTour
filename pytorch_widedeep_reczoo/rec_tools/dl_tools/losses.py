import torch
import torch.nn as nn


class SigmoidBoundedRMSELoss(nn.Module):
    def __init__(self, high: float = 5.0, low: float = 1.0):
        super().__init__()
        self.high = high
        self.low = low

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bounded_preds = torch.sigmoid(input) * (self.high - self.low) + self.low
        loss = (bounded_preds - target) ** 2
        return torch.sqrt(torch.mean(loss))
