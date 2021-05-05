import unittest
import torch
from loss import loss


class LossTest(unittest.TestCase):
    def setUp(self):
        target = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=torch.float32)

        self.target = torch.cat(5*[target]).reshape((5, 3, 3))
        return super().setUp()

    def test_dice_loss_with_gt_inut(self):

        gt_input = torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        gt_input = torch.cat(5*[gt_input]).reshape((5, 3, 3))
        dice_loss = loss.dice_loss(
            gt_input, self.target, apply_sigmoid=False, smooth=1)
        self.assertListEqual(
            list(dice_loss.numpy()),
            [0.0] * 5
        )

    def test_dice_loss_with_wrong_input(self):
        fully_wrong_input = torch.tensor([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ])
        fully_wrong_input = torch.cat(5*[fully_wrong_input]).reshape((5, 3, 3))
        dice_loss = loss.dice_loss(
            fully_wrong_input, self.target, apply_sigmoid=False, smooth=1)
        for i, j in zip(list(dice_loss.numpy()), [0.9] * 5):
            self.assertAlmostEqual(i, j)


if __name__ == '__main__':
    unittest.main()
