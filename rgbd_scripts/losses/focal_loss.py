import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
        Calculates focal loss by each y_pred map from batch and returns the average.
        Note, that y_true is expected to be Binary

        Paper: https://arxiv.org/pdf/2006.14822.pdf

        Implementation inspired by
        https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/comments
        and https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/focal.py

        Args:
            apply_sigmoid: set False if sigmoid is already applied to model y_pred
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error

    '''

    def __init__(
        self,
        apply_sigmoid: bool = True,
        alpha=0.8,
        gamma=2.0
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,):
        batch_size = y_pred.shape[0]
        if self.apply_sigmoid:
            CE = F.binary_cross_entropy_with_logits
        else:
            CE = F.binary_cross_entropy

        y_pred = y_pred.view(batch_size, 1, -1)
        y_true = y_true.view(batch_size, 1, -1)

        logp = CE(y_pred, y_true, reduction='none')
        p = torch.exp(-logp)
        focal_loss = self.alpha * (1-p)**self.gamma * logp
        return focal_loss.mean()
