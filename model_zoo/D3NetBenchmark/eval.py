# pytorch
from numpy.lib import utils
import torch
import torchvision
from torch.utils.data import DataLoader

# general
import os
import cv2
import numpy as np
from tqdm import tqdm

# mine
import my_custom_transforms as mtr
from dataloader_rgbdsod import RgbdSodDataset
from PIL import Image
from model.RgbNet import MyNet as RgbNet
from model.RgbdNet import MyNet as RgbdNet
from model.DepthNet import MyNet as DepthNet
from utils import f_measure

size = (224, 224)
save_eval_images = False
datasets_path = 'datasets/TestingSet/'
pretrained_model_path = 'eval/D3Net/'
test_datasets = ['STERE']
pretrained_models = {
    'RgbNet': pretrained_model_path + 'RgbNet.pth',
    'RgbdNet': pretrained_model_path + 'RgbdNet.pth',
    'DepthNet': pretrained_model_path + 'DepthNet.pth'
    }
result_path = 'eval/D3Net/result/'
os.makedirs(result_path, exist_ok=True)

for tmp in ['D3Net']:
    os.makedirs(os.path.join(result_path, tmp), exist_ok=True)
    for test_dataset in test_datasets:
        os.makedirs(os.path.join(result_path, tmp, test_dataset), exist_ok=True)

model_rgb = RgbNet().cuda()
model_rgbd = RgbdNet().cuda()
model_depth = DepthNet().cuda()

model_rgb.load_state_dict(torch.load(pretrained_models['RgbNet'])['model'])
model_rgbd.load_state_dict(torch.load(pretrained_models['RgbdNet'])['model'])
model_depth.load_state_dict(torch.load(pretrained_models['DepthNet'])['model'])

model_rgb.eval()
model_rgbd.eval()
model_depth.eval()

transform_test = torchvision.transforms.Compose(
    [
        mtr.Resize(size),
        mtr.ToTensor(),
        mtr.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], elems_do=['img']
            )
    ])

test_loaders = []
for test_dataset in test_datasets:
    val_set = RgbdSodDataset(datasets_path + test_dataset, transform=transform_test)
    test_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True))

for index, test_loader in enumerate(test_loaders):
    dataset = test_datasets[index]
    print('Test [{}]'.format(dataset))
    mae = 0
    f_score = 0
    for i, sample_batched in enumerate(tqdm(test_loader)):
        input, gt = model_rgb.get_input(sample_batched), model_rgb.get_gt(sample_batched)
        # gt.shape torch.Size([1, 1, 224, 224])
        # input[0].shape: torch.Size([1, 3, 224, 224]),input[1].shape: torch.Size([1, 1, 224, 224]) -- input is RGB+Depth image

        with torch.no_grad():
            output_rgb = model_rgb(input)  # all have shape ([1, 1, 224, 224])
            output_rgbd = model_rgbd(input)
            output_depth = model_depth(input)

        result_rgb = model_rgb.get_result(output_rgb)  # have shape (224, 224)
        result_rgbd = model_rgbd.get_result(output_rgbd)
        result_depth = model_depth.get_result(output_depth)  # img = Image.fromarray(result_depth, 'L').show() -- some noises

        id = sample_batched['meta']['id'][0]
        gt_src = np.array(Image.open(sample_batched['meta']['gt_path'][0]).convert('L'))  # ground truth picture from dataset

        result_rgb = (cv2.resize(result_rgb, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)  # interpolate small picture -> big. But why x255?
        result_rgbd = (cv2.resize(result_rgbd, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
        result_depth = (cv2.resize(result_depth, gt_src.shape[::-1], interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
        # now they contains real images look like mask from gt

        ddu_mae = np.mean(np.abs(result_rgbd/255.0 - result_depth/255.0))  # compare original sizes
        result_d3net = result_rgbd if ddu_mae < 0.15 else result_rgb  # some magic choice

        if save_eval_images:
            Image.fromarray(result_d3net).save(os.path.join(result_path, 'D3Net', dataset, id + '.png'))

        result_d3net = result_d3net.astype(np.float64)/255.0  # why ?
        gt_src = gt_src/255.0

        mae += np.mean(np.abs(result_d3net - gt_src))
        f_score += f_measure(result_d3net, gt_src)  # f_score.shape -> torch.Size([255])

    print(f'MAE = {mae/len(test_loader)}, F-Score = { (f_score/len(test_loader)).max().item() }')  # why so for f_score?



