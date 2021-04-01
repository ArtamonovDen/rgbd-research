import argparse
import os

from torch.utils import data
from rgbd_scripts import metrics
from data import test_dataset
from models.BBSNet_model import BBSNet
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')


def init_wnb_config():
    return dict(
        wandb_project='BBS-Net-evaluation',
        model_name='BBS'
    )


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str,
                    default='./datasets/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# load the model
model = BBSNet()
# Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load('./eval/BBS-Net/BBSNet_epoch_best.pth'))
model.cuda()
model.eval()

# test
test_datasets = ['STERE', 'NJU2K_TEST', 'NLPR_TEST',
                 'DES', 'SSD', 'LFSD', 'SIP']
config = init_wnb_config()
with wandb.init(project=config['wandb_project'], config=config):
    for dataset in test_datasets:
        print(f'Processing {dataset}')
        if not os.path.exists(dataset_path + dataset):
            print(f'Dataset {dataset_path + dataset} does not exist')
            continue
        total_f1, total_mae = 0.0, 0.0
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        depth_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            _, res = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear',
                            align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae = metrics.mae(res, gt)
            f1 = metrics.f_beta_measure(res, gt)
            wandb.log({f'mae_{dataset}': mae, f'f1{dataset}': f1})
            wandb.log({'res': [wandb.Image(torch.tensor(res))], 'gt': [wandb.Image(torch.tensor(gt))]})
            total_mae += mae
            total_f1 += f1
        print(f'Average f1 is {total_f1 / test_loader.size}')
        print(f'Average mae is {total_mae / test_loader.size}')
    print('Test Done!')
