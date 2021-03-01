
import wandb
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from enum import Enum
from torch.utils.data import DataLoader

from model.DepthNet import MyNet as DepthNetModel
from model.RgbdNet import MyNet as RgbNetModel
from model.RgbNet import MyNet as RgbdNetModel

import my_custom_transforms as mtr
import utils
from dataloader_rgbdsod import RgbdSodDataset


class D3NetModels(Enum):
    DepthNet = 'DepthNet'
    RgbNet = 'RgbNet'
    RgbdNet = 'RgbdNet'


config = dict(
    model_name='D3Net',
    backbone_path='./model/vgg16_feat.pth',
    D3Net_model=D3NetModels.DepthNet.value,
    data_normalize_mean=[0.485, 0.456, 0.406],
    data_normalize_std=[0.229, 0.224, 0.225],
    data_size=(224, 224),
    batch_size=1,
    epochs=10,
    train_datasets=['./datasets/NJU2K_TRAIN'],
    val_datasets=['./datasets/NJU2K_TEST'],
    val_interval=5,
    optimizer='Adam',
    optimizer_params={'weight_decay': 0, 'betas': [0.9, 0.99]},
    scheduler_type='Constant',
    scheduler_params={}
    )

config['learning_rate'] = 1.25e-5*(config['batch_size'])


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


def make_model(model_name, backbone_path):

    if model_name == D3NetModels.DepthNet.value:
        return DepthNetModel(backbone_path)
    if model_name == D3NetModels.RgbNet.value:
        return RgbNetModel(backbone_path)
    if model_name == D3NetModels.RgbdNet.value:
        return RgbdNetModel(backbone_path)


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({'epoch': epoch, 'loss': loss}, step=example_ct)
    print(f'Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}')


def test(model, criterion, val_loader, device):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        loss_total = 0
        mae_avg, f_score_avg = 0, 0
        tbar = tqdm(val_loader)
        for i, sample_batched in enumerate(tbar):
            rgb, dep, gt = sample_batched['img'].to(device), sample_batched['depth'].to(device), sample_batched['gt'].to(device)
            output = model((rgb, dep))
            loss = criterion(output, gt)
            loss_total += loss.item()

            result = model.get_result(output)
            mae, f_score = utils.get_metric(sample_batched, result)
            mae_avg, f_score_avg = mae_avg + mae, f_score_avg + f_score

            wandb.log({'loss': loss, 'mae': mae, 'f_score': f_score.max().item()}, step=i)

        print('Loss: %.3f' % (loss_total / (i + 1)))
        mae_avg, f_score_avg = mae_avg/len(tbar), f_score_avg/len(tbar)
        print(f'mae:{mae_avg:.4f} f_max:{f_score_avg.max().item():.4f}')
        wandb.log({'aveloss': (loss_total / (i + 1)), 'ave_mae': mae_avg, 'f_score_ave': f_score_avg.max().item()}, step=i)

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, sample_batched, "d3net_model.onnx")
    # wandb.save("d3net_model.onnx")


def model_pipeline(hyperparameters):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # config = wandb.config
        model = make_model(config['D3Net_model'], config['backbone_path'])
        model = model.to(device)

        train_loader = make_train_data_loader(config)
        val_loaders = make_val_data_loader(config)

        optimizer = utils.get_optimizer(config['optimizer'], model.parameters(), config['learning_rate'], config['optimizer_params'])
        scheduler = utils.get_scheduler(config['scheduler_type'], optimizer, config['scheduler_params'])
        criterion = nn.BCELoss()  # TODO should we use .cuda() ?

        # Train
        print('Start training')
        wandb.watch(model, criterion, log="all", log_freq=10)
        example_ct = 0
        for epoch in tqdm(range(config['epochs'])):
            loss_total = 0
            model.train()
            for i, sample_batched in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                rgb, dep, gt = sample_batched['img'].to(device), sample_batched['depth'].to(device), sample_batched['gt'].to(device)
                output = model((rgb, dep))
                loss = criterion(output, gt)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()
                example_ct += len(sample_batched)

                if ((i+1) % 25) == 0:
                    train_log(loss, example_ct, epoch)

            scheduler.step()

        # Test
        print('Start testing')
        test(model, criterion, val_loaders[0], device)

        # Save model
        print('Save model')
        snapshot_path = 'snapshot_[{}]_[{}]'.format(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())), config['D3Net_model'])
        os.makedirs(snapshot_path)
        pth_state = {
            'current_epoch': 0,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(pth_state, os.path.join(snapshot_path, f'best_{config["D3Net_model"]}.pth'))

    return model


if __name__ == '__main__':
    model_pipeline(config)
