import unittest
import torch
from BBSNet.models.BBSNet_model_effnet import BBSNet as BBSNetEffnet


class ModelTest(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_bbsnet_with_efficientnet(self):
        x = torch.randn((2, 3, 224, 224))
        x_depth = torch.randn((2, 1, 224, 224))

        model = BBSNetEffnet()
        model.eval()
        s1, s2 = model(x, x_depth)

        self.assertListEqual(list(s1.shape), [2, 1, 224, 224])
        self.assertListEqual(list(s2.shape), [2, 1, 224, 224])


if __name__ == '__main__':
    unittest.main()
