import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    '''
        Calculates loss by each y_pred map from batch and returns the average.
        Note, that y_true is expected to be Binary

        Paper: https://arxiv.org/pdf/2006.14822.pdf

        Implementation inspired by
        https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/c3f7449ee510ad154ef9dc08d2f2443f0854e1d5/loss_functions.py#L79
        and https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/dice.py

        Args:
            apply_sigmoid: set False if sigmoid is already applied to model y_pred
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error

    '''

    def __init__(
        self,
        apply_sigmoid: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7
    ):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,):
        batch_size = y_pred.shape[0]
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)  # TODO F.logsigmoid(y_pred).exp() ?

        y_pred = y_pred.view(batch_size, 1, -1)
        y_true = y_true.view(batch_size, 1, -1)

        intersection = torch.sum(y_pred * y_true, dim=2)
        cardinality = torch.sum(y_pred + y_true, dim=2)

        score = (2. * intersection + self.smooth) / \
            (cardinality + self.smooth).clamp(min=self.eps)
        return (1 - score).mean()
