import torch


@DeprecationWarning
def batched_dice_loss(input, target, apply_sigmoid=True, smooth=1):
    '''
        Dice Loss function
        https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/c3f7449ee510ad154ef9dc08d2f2443f0854e1d5/loss_functions.py#L79
        https://arxiv.org/pdf/2006.14822.pdf

        Args:
            target: a tensor of shape [1, H, W].
            input: a tensor of shape [C, H, W]. Corresponds to output of the model.
            apply_sigmoid: set False if sigmoid is already applied to model output
            smooth: TODO
        Returns:
            dice_loss: the Sørensen–Dice loss.
    '''

    batch_size = input.shape[0]
    if apply_sigmoid:
        input = torch.sigmoid(input)

    input = input.view(batch_size, -1)
    target = target.view(batch_size, -1)
    intersection = (input * target).sum(1)
    score = (2. * intersection + smooth)/(input.sum(1) + target.sum(1) + smooth)
    return 1 - score


def dice_loss(input: torch.tensor, target: torch.tensor, apply_sigmoid=True, smooth=1):
    '''
        Dice Loss function
        https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/c3f7449ee510ad154ef9dc08d2f2443f0854e1d5/loss_functions.py#L79
        https://arxiv.org/pdf/2006.14822.pdf

        Args:
            target: a tensor of shape [N, *].
            input: a tensor of shape [N, *]. Corresponds to batched output of the model.
            apply_sigmoid: set False if sigmoid is already applied to model output
            smooth: TODO
        Returns:
            dice_loss: the Sørensen–Dice loss.
    '''
    if apply_sigmoid:
        input = torch.sigmoid(input)

    input = input.flatten()
    target = target.flatten()
    intersection = (input * target).sum()
    score = (2. * intersection + smooth)/(input.sum() + target.sum() + smooth)
    return 1 - score
