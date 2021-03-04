import wandb
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from enum import Enum

from model.DepthNet import MyNet as DepthNetModel
from model.RgbdNet import MyNet as RgbNetModel
from model.RgbNet import MyNet as RgbdNetModel

import utils
from utils import make_val_data_loader, make_train_data_loader


class D3NetModels(Enum):
    DepthNet = 'DepthNet'
    RgbNet = 'RgbNet'
    RgbdNet = 'RgbdNet'


config = dict(
    wandb_project='D3Net-project',
    model_name='D3Net',
    backbone_path='./model/vgg16_feat.pth',
    D3Net_model='DepthNet',
    data_normalize_mean=[0.485, 0.456, 0.406],
    data_normalize_std=[0.229, 0.224, 0.225],
    data_size=(224, 224),
    batch_size=8,
    epochs=32,
    train_datasets=['./datasets/NJU2K_TRAIN'],
    val_datasets=['./datasets/NJU2K_TEST'],
    val_interval=4,
    optimizer='Adam',
    optimizer_params={'weight_decay': 0, 'betas': [0.9, 0.99]},
    scheduler_type='Constant',
    scheduler_params={}
    )

config['learning_rate'] = 1.25e-5*(config['batch_size'])


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

        loss_avg, mae_avg, f_score_avg = loss_total / (i + 1), mae_avg/len(tbar), (f_score_avg/len(tbar)).max().item()
        print(f'loss: {loss_avg:.3f} mae:{mae_avg:.4f} f_max:{f_score_avg:.4f}')
        wandb.log({'val_ave_loss': (loss_avg), 'val_ave_mae': mae_avg, 'val_f_score_ave': f_score_avg})

    # Save the model in the exchangeable ONNX format
    # torch.onnx.export(model, sample_batched, "d3net_model.onnx")
    # wandb.save("d3net_model.onnx")

    return mae_avg, f_score_avg


def model_pipeline(config):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_metrics = None
    snapshot_path = 'snapshot_[{}]_[{}]'.format(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())), config['D3Net_model'])
    os.makedirs(snapshot_path)

    with wandb.init(project=config['wandb_project'], config=config):
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

            if ((epoch+1) % config['val_interval']) == 0:
                print(f'Start validation on {epoch+1} epoch')
                mae, f_score = test(model, criterion, val_loaders[0], device)
                if (not best_metrics) or (best_metrics and mae < best_metrics[0]):
                    best_metrics = (mae, f_score)
                    print(f'Dest metrics were updaeted: mae:{mae:.4f} f_max:{f_score:.4f}')
                    print('Save best model')
                    pth_state = {
                        'best_metric': best_metrics,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }
                    torch.save(pth_state, os.path.join(snapshot_path, f'best_{config["D3Net_model"]}.pth'))

        # Test
        print('Final test')
        mae, f_score = test(model, criterion, val_loaders[0], device)
        if (not best_metrics) or (best_metrics and mae < best_metrics[0]):
            best_metrics = (mae, f_score)
            print('Save best model')
            pth_state = {
                'best_metric': best_metrics,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(pth_state, os.path.join(snapshot_path, f'best_{config["D3Net_model"]}.pth'))

    return model


if __name__ == '__main__':
    model_pipeline(config)
