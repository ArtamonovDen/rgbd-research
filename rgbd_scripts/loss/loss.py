import torch


def dice_loss(input, target, apply_sigmoid=True, smooth=1):
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

    # TODO Add batch support

    if apply_sigmoid:
        input = torch.sigmoid(input)

    input = torch.flatten(input)
    target = torch.flatten(target)
    intersection = (input * target).sum()
    score = (2. * intersection + smooth)/(input.sum() + target.sum() + smooth)
    return 1 - score
