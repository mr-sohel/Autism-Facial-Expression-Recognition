import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(torch.float32)
            else:
                self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_fn(loss_type, class_weights=None):
    if class_weights is not None:
        class_weights = class_weights.to(torch.float32)

    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_type == "ce_smooth":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    elif loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
