import unittest
import torch
from loss import loss


class LossTest(unittest.TestCase):
    def setUp(self):
        self.output_shape = (1, 1, 3, 3)

        self.target = self.make_batch(torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=torch.float32))

        return super().setUp()

    def make_batch(self, input):
        return torch.cat(self.output_shape[0] * [input]).reshape(self.output_shape)

    def test_dice_loss_with_gt_inut(self):

        gt_input = self.make_batch(torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]))
        dice_loss = loss.dice_loss(
            gt_input, self.target, apply_sigmoid=False, smooth=1)
        self.assertAlmostEqual(0, dice_loss.item())

    def test_dice_loss_with_wrong_input(self):
        fully_wrong_input = self.make_batch(torch.tensor([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]))
        dice_loss = loss.dice_loss(
            fully_wrong_input, self.target, apply_sigmoid=False, smooth=1)  # more the batch size, more loss: smooth is applied over a batch
        self.assertAlmostEqual(0.9, dice_loss.item())


if __name__ == '__main__':
    unittest.main()
