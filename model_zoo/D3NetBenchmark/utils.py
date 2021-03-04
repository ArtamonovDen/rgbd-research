import torch
import random
import torchvision
import os
import my_custom_transforms as mtr
from dataloader_rgbdsod import RgbdSodDataset
from torch.utils.data import DataLoader
########################################[ Optimizer ]########################################


def get_optimizer(mode, train_params, lr, kwargs):
    opt_default = {
        'SGD': {'momentum': 0.9, 'weight_decay': 5e-4, 'nesterov': False},
        # 'Adam' : {'weight_decay':5e-4, 'betas':[0.9, 0.99]},
        'Adam': {'weight_decay': 0, 'betas': [0.9, 0.99]},
        'TBD': {}
        }

    for k in opt_default[mode].keys():
        if k not in kwargs.keys():
            kwargs[k] = opt_default[mode][k]

    if mode == 'SGD':
        return torch.optim.SGD(train_params, lr=lr, **kwargs)
    elif mode == 'Adam':
        return torch.optim.Adam(train_params, lr=lr, **kwargs)


########################################[ Scheduler ]########################################
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, epoch_max, power=0.9, last_epoch=-1, cutoff_epoch=1000000):
        self.epoch_max = epoch_max
        self.power = power
        self.cutoff_epoch = cutoff_epoch
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch<self.cutoff_epoch:
            return [  base_lr *   ( 1-( 1.0*self.last_epoch / self.epoch_max) )**self.power  for base_lr in self.base_lrs   ]
        else:
            return [  base_lr *   ( 1-( 1.0*self.cutoff_epoch / self.epoch_max) )**self.power  for base_lr in self.base_lrs   ]

class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [  base_lr * 1.0  for base_lr in self.base_lrs   ]


def get_scheduler(mode, optimizer, kwargs):
    if mode=='Poly':
        return  PolyLR(optimizer, **kwargs)
    elif mode=='Constant':
        return  ConstantLR(optimizer, **kwargs)
    elif mode=='Step':
        return  torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif mode=='MultiStep':
        return  torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)

########################################[ Evaluator ]########################################
import numpy as np
from PIL import Image
import cv2

def eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def f_measure(pred, gt):
    beta2 = 0.3
    with torch.no_grad():
        pred = torch.from_numpy(pred).float().cuda()
        gt = torch.from_numpy(gt).float().cuda()

        prec, recall = eval_pr(pred, gt, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
    return f_score


def get_metric(sample_batched, result, result_save_path=None, if_recover=True):
    id = sample_batched['meta']['id'][0]
    gt = np.array(Image.open(sample_batched['meta']['gt_path'][0]).convert('L'))/255.0

    if if_recover:
        result = cv2.resize(result, gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
    else:
        gt = cv2.resize(gt, result.shape[::-1], interpolation=cv2.INTER_NEAREST)

    result = (result*255).astype(np.uint8)

    if result_save_path is not None:
        Image.fromarray(result).save(os.path.join(result_save_path, id+'.png'))

    result = result.astype(np.float64)/255.0

    mae = np.mean(np.abs(result-gt))
    f_score = f_measure(result, gt)
    return mae, f_score

def metric_better_than(metric_a, metric_b):
    if metric_b is None:
        return True
    if isinstance(metric_a,list) or isinstance(metric_a,np.ndarray):
        metric_a,metric_b=metric_a[0],metric_b[0]
    return metric_a < metric_b


########################################[ Loss ]########################################




########################################[ Random ]########################################
# 固定随机种子！
def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# #保存记录点
# def save_checkpoint(state, is_best, path, filename, if_save_checkpoint=False):
#     if if_save_checkpoint:
#         torch.save(state, os.path.join(path, 'checkpoint.pth'))
#     if is_best:
#         torch.save(state, os.path.join(path, 'best.pth'))


########################################[ Data Loaders ]########################################

def make_train_data_loader(config):
    transform_train = torchvision.transforms.Compose([
        mtr.RandomFlip(),
        mtr.Resize(tuple(config['data_size'])),
        mtr.ToTensor(),
        mtr.Normalize(config['data_normalize_mean'], config['data_normalize_std'], elems_do=['img']),
    ])

    # TODO maxnum ?, num_workers ?

    train_set = RgbdSodDataset(datasets=config['train_datasets'], transform=transform_train, max_num=0, if_memory=False)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    return train_loader


def make_val_data_loader(config):
    transform_val = torchvision.transforms.Compose([
        mtr.Resize(tuple(config['data_size'])),
        mtr.ToTensor(),
        mtr.Normalize(config['data_normalize_mean'], config['data_normalize_std'], elems_do=['img']),
    ])

    val_loaders = list()

    for val_dataset in config['val_datasets']:
        val_set = RgbdSodDataset(val_dataset, transform=transform_val, max_num=0, if_memory=False)
        val_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True))

    return val_loaders
