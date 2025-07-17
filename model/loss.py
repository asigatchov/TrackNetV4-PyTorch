import torch
import torch.nn as nn


class WeightedBinaryCrossEntropy(nn.Module):
    """
    Weighted Binary Cross Entropy Loss as defined in the paper
    WBCE = -Σ[(1-w)² * ŷ * log(y) + w² * (1-ŷ) * log(1-y)]
    where w = y (prediction value as weight)
    """

    def __init__(self, epsilon=1e-7):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Model predictions [B, 3, H, W], range [0,1]
            y_true: Ground truth labels [B, 3, H, W], range {0,1}
        Returns:
            loss: Scalar loss value
        """
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        w = y_pred
        term1 = (1 - w) ** 2 * y_true * torch.log(y_pred)
        term2 = w ** 2 * (1 - y_true) * torch.log(1 - y_pred)
        wbce = -(term1 + term2)
        return wbce.mean()
