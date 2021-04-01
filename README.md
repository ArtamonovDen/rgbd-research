# rgbd-research

This repo aims to collect and compare several RGB-D models.

Models in model zoo:

- [D3Net](https://github.com/DengPingFan/D3NetBenchmark), [arxiv](https://arxiv.org/pdf/1907.06781.pdf)
- [BBS-Net](https://github.com/zyjwuyan/BBS-Net), [arxiv](https://arxiv.org/pdf/2007.02713v2.pdf)
- [UCNet](https://github.com/JingZhang617/UCNet), [arxiv](https://arxiv.org/abs/2009.03075)


## Models Evaluation

### D3Net

Run evaluation script with calcualtion F score and MAE

| Dataset | MAE   | F-Score |
|---------|-------|---------|
| SSD     | 0.062 | 0.82    |
| DES     | 0.03  | 0.89    |
| STERE   | 0.048 | 0.89    |

### D3Net with pytorch [VGG-16 backbone](https://pytorch.org/vision/0.8/models.html#torchvision.models.vgg16)

| Dataset     | MAE    | F-Score |
|-------------|--------|---------|
| SSD         | 0.987  | 0.787   |
| DES         | 0.0614 | 0.822   |
| STERE       | 0.1    | 0.792   |
| NJU2K_TRAIN | 0.07   | 0.872   |
| NJU2K_TEST  | 0.0869 | 0.836   |

### BBS-Net

| Dataset    | MAE   | F-Score |
|------------|-------|---------|
| NJU2K_TEST | 0.036 | 0.903   |
| NLPR_TEST  | 0.042 | 0.835   |
| DES        | 0.02  | 0.913   |
| SSD        | 0.047 | 0.837   |
| LFSD       | 0.073 | 0.846   |
| SIP        | 0.047 | 0.869   |
