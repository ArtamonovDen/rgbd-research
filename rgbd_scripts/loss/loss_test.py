import unittest
import torch
from loss import loss


class LossTest(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=torch.float32)
        return super().setUp()

    def test_dice_loss_with_gt_inut(self):

        gt_input = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        dice_loss = loss.dice_loss(gt_input, self.target, apply_sigmoid=False, smooth=1)
        self.assertAlmostEqual(dice_loss.item(), 0.0)

    def test_dice_loss_with_wrong_input(self):
        fully_wrong_input = torch.tensor([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ])
        dice_loss = loss.dice_loss(fully_wrong_input, self.target, apply_sigmoid=False, smooth=1)
        self.assertAlmostEqual(dice_loss.item(), 0.9)


if __name__ == '__main__':
    unittest.main()
