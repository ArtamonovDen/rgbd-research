import unittest
import torch
from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss


class LossTest(unittest.TestCase):
    def setUp(self):
        self.output_shape = (3, 1, 3, 3)

        self.target = self.make_batch(torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=torch.float32))

        self.gt_input = self.make_batch(torch.tensor([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=torch.float32))

        self.fully_wrong_input = self.make_batch(torch.tensor([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ], dtype=torch.float32))

        self.just_input = self.make_batch(torch.tensor([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=torch.float32))

        self.just_input2 = self.make_batch(torch.tensor([
            [0.2, 1., 0],
            [0.5, 1., 0.2],
            [0, 0, 1],
        ], dtype=torch.float32))

        return super().setUp()

    def make_batch(self, input):
        return torch.cat(self.output_shape[0] * [input]).reshape(self.output_shape)

    def test_dice_loss_with_gt_inut(self):
        dice_loss = DiceLoss(apply_sigmoid=False, smooth=1)
        ret_loss = dice_loss(self.gt_input, self.target)
        self.assertAlmostEqual(0, ret_loss.item())

    def test_dice_loss_with_wrong_input(self):
        dice_loss = DiceLoss(apply_sigmoid=False, smooth=0.0)
        ret_loss = dice_loss(self.fully_wrong_input, self.target)
        self.assertAlmostEqual(1.0, ret_loss.item())

    def test_dice_loss(self):
        dice_loss = DiceLoss(apply_sigmoid=False, smooth=0.0)
        ret_loss = dice_loss(self.just_input, self.target)
        print(ret_loss)
        self.assertAlmostEqual(0.25, ret_loss.item())

    def test_focal_loss_with_gt_inut(self):
        loss = FocalLoss(apply_sigmoid=False)
        ret_loss = loss(self.gt_input, self.target)
        self.assertAlmostEqual(0, ret_loss.item())

    def test_focal_loss_with_wrong_inut(self):
        loss = FocalLoss(apply_sigmoid=False)
        ret_loss = loss(self.fully_wrong_input, self.target)
        self.assertAlmostEqual(80, ret_loss.item())

    def test_focal_loss(self):
        loss = FocalLoss(apply_sigmoid=False)
        ret_loss = loss(self.just_input, self.target)
        self.assertEqual(17, int(ret_loss.item()))


if __name__ == '__main__':
    unittest.main()
