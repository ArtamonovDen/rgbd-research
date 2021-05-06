import unittest
import torch
import torch.nn as nn
from BBSNet.models.BBSNet_model_effnet import BBSNet as BBSNetEffnet


class BBSModelTest(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_efficientnet_backbone(self):
        x = torch.randn((2, 3, 224, 224))
        x_depth = torch.randn((2, 1, 224, 224))

        model = BBSNetEffnet()
        model.eval()
        s1, s2 = model(x, x_depth)
        self.assertListEqual(list(s1.shape), [2, 1, 224, 224])
        self.assertListEqual(list(s2.shape), [2, 1, 224, 224])

    def test_efficientnet_backbone_with_loss(self):

        CE_loss = nn.BCEWithLogitsLoss()

        x = torch.randn((2, 3, 224, 224))
        gts = torch.randn((2, 1, 224, 224))
        x_depth = torch.randn((2, 1, 224, 224))

        model = BBSNetEffnet()
        s1, s2 = model(x, x_depth)
        loss1 = CE_loss(s1, gts)
        loss2 = CE_loss(s2, gts)
        loss = loss1+loss2
        loss.backward()
        self.assertTrue(True)  # Check that backward has no errors


if __name__ == '__main__':
    unittest.main()
