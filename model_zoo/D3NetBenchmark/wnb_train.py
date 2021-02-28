
import wandb
import enum
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from model.DepthNet import MyNet as DepthNetModel
from model.RgbdNet import MyNet as RgbNetModel
from model.RgbNet import MyNet as RgbdNetModel

import my_custom_transforms as mtr
import utils
from dataloader_rgbdsod import RgbdSodDataset

class D3NetModels(enum):
    DepthNet = 'DepthNet'
    RgbNet = 'RgbNet'
    RgbdNet = 'RgbdNet'


config = dict(
    model_name='D3Net',
    backbone_path='./model/vgg16_feat.pth',
    D3Net_model=D3NetModels.DepthNet,
    data_normalize_mean=[0.485, 0.456, 0.406],
    data_normalize_std=[0.229, 0.224, 0.225],
    data_size=(224, 224),
    batch_size=128,
    train_datasets=['NJU2K_TRAIN'],
    val_datasets=['NJU2K_TRAIN'],
    optimizer='Adam',
    optimizer_params={'weight_decay': 0, 'betas': [0.9, 0.99]},
    scheduler_type='Constant',
    scheduler_params={}
    )

config['learning_rate'] = 1.25e-5*(config['train_datasets'])


def make_train_data_loader(config):
    transform_train = torchvision.transforms.Compose([
        mtr.RandomFlip(),
        mtr.Resize(config.data_size),
        mtr.ToTensor(),
        mtr.Normalize(config.data_normalize_mean, config.data_normalize_std, elems_do=['img']),
    ])

    # TODO maxnum ?, num_workers ?

    train_set = RgbdSodDataset(datasets=config.train_datasets, transform=transform_train, max_num=0, if_memory=False)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader


def make_val_data_loader(config):
    transform_val = torchvision.transforms.Compose([
        mtr.Resize(config.data_size),
        mtr.ToTensor(),
        mtr.Normalize(config.data_normalize_mean, config.data_normalize_std, elems_do=['img']),
    ])

    val_loaders = list()

    for val_dataset in config.val_datasets:
        val_set = RgbdSodDataset(val_dataset, transform=transform_val, max_num=0, if_memory=False)
        val_loaders.append(DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True))

    return val_loaders


def make_model(model_name, backbone_path):

    if model_name == D3NetModels.DepthNet:
        return DepthNetModel(backbone_path)
    if model_name == D3NetModels.RgbNet:
        return RgbNetModel(backbone_path)
    if model_name == D3NetModels.RgbdNet:
        return RgbdNetModel(backbone_path)


def model_pipeline(hyperparameters):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with wandb.init(project="pytorch-demo", config=hyperparameters):
        config = wandb.config
        model = make_model(config.D3Net_model, config.backbone_path).to(device)

        train_loader = make_train_data_loader(config)
        val_loaders = make_val_data_loader(config)

        optimizer = utils.get_optimizer(config.optimizer, model.parameters(), config.learning_rate, config.optimizer_params)
        scheduler = utils.get_scheduler(config.scheduler_type, optimizer, config.scheduler_params)

        # TODO firstly criterion ?

        criterion = nn.BCELoss()  # TODO should we use .cuda() ?


        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

      # and use them to train the model
        train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
        test(model, test_loader)

    return model


if __name__ == '__main__':
    pass
