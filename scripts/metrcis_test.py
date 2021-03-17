import unittest
import numpy as np
import metrics as m


class MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        self.pred = np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 0]
        ])
        self.tp = 3
        self.true_precision = 3 / 4
        self.true_recall = 3 / 5
        self.true_f_score = 2 * self.true_precision * self.true_recall / (self.true_precision + self.true_recall)

        return super().setUp()

    def test_precision_recall(self):
        p, r = m.precision_recall(self.pred, self.gt)
        self.assertEqual(p, self.true_precision)
        self.assertEqual(r, self.true_recall)

    def test_f_score(self):
        self.assertEqual(m.f_beta_measure(self.pred, self.gt, beta=1), self.true_f_score)

    def test_f_score_with_threshold(self):
        noisy_pred = np.array([
            [0, 0, 0.2],
            [0.4, 0.8, 0],
            [0, 0.9, 0]
        ])
        noisy_pred_threshold_0_5 = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]
        ])
        noisy_pred_threshold_0 = np.ones((3, 3))
        f = m.f_beta_score_with_threshold(noisy_pred, self.gt, 3, beta=1)
        self.assertEqual(list(f.keys()), [0, 0.5, 1])
        self.assertEqual(f[1], 0)
        self.assertEqual(f[0.5], m.f_beta_measure(noisy_pred_threshold_0_5, self.gt, beta=1))
        self.assertEqual(f[0], m.f_beta_measure(noisy_pred_threshold_0, self.gt, beta=1))

        print(f)


if __name__ == '__main__':
    unittest.main()
