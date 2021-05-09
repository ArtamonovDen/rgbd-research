import argparse
import logging
import os
import sys
import torch
import wandb
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime


from model_zoo.BBSNet.models.BBSNet_model import BBSNet
from model_zoo.BBSNet.models.BBSNet_model_effnet import BBSNet as BBSNet_effnet
from model_zoo.BBSNet import bbs_utils, data
from loss import loss

MODELS = {
    'BBS-Net': BBSNet,
    'BBS-Net-Effnet': BBSNet_effnet
}

LOSSES = {
    'cross-entropy': nn.BCEWithLogitsLoss(),
    'dice': loss.dice_loss
}


def init_wnb_config(args_):
    return dict(
        wandb_project=args_.wandb_project,
        model_name=args_.model,
        epochs=args_.epoch,
        lr=args_.lr,
        input_size=args_.input_size,
        batch_size=args_.batch_size,
        clip=args_.clip,
        decay_rate=args_.decay_rate,
        decay_epoch=args_.decay_epoch
    )


def validate(model, val_dataset, device, epoch):
    model.eval()
    mae_sum = 0.0
    with torch.no_grad():
        for i in range(val_dataset.size):
            image, gt, depth, name, img_for_post = val_dataset.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image, depth = image.to(device), depth.to(device)
            _, res = model(image, depth)

            res = F.upsample(res, size=gt.shape,
                             mode='bilinear', align_corners=False)  # TODO move to separate method
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8) # TODO why?

            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

        mae = mae_sum/val_dataset.size
        wandb.log({'test MAE': torch.tensor(mae)})
        logging.info(f'Validation. Epoch {epoch}. MAE: {mae}')

        return mae


def update_best_mae(val_mae, best_mae, best_epoch, model, save_path, model_name, epoch):
    if epoch == 0:
        best_mae = val_mae
        best_epoch = 0
        logging.info(f'Init best MAE: {best_mae}')
    elif val_mae < best_mae:
        best_mae = val_mae
        best_epoch = epoch
        logging.info(
            f'Best MAE was updated. Best MAE: {best_mae}. Best epoch: {best_epoch}. Saving best model')
        torch.save(model.state_dict(), save_path + f'{model_name}_best.pth')

    return best_mae, best_epoch


def log_step(loss1, loss2, loss, cur_step, total_step, cur_epoch):
    logging.info(
        f'Epoch {cur_epoch}. Step {cur_step}/{total_step}], Loss1: {loss1} Loss2: {loss2} Loss: {loss}')
    wandb.log(
        {'loss1': loss1, 'loss2': loss2, 'loss': loss})


def train_epoch(model, cur_epoch, optimizer, train_dataloader, device, loss_f, clip_grad):

    model.train()
    loss_over_epoch = 0.0

    for i, (images, gts, depths) in enumerate(train_dataloader):
        optimizer.zero_grad()
        images, gts, depths = images.to(device), gts.to(
            device), depths.to(device)  # TODO Make in dataloader
        s1, s2 = model(images, depths)
        loss1 = loss_f(s1, gts)
        loss2 = loss_f(s2, gts)
        loss = loss1+loss2

        loss.backward()
        bbs_utils.clip_gradient(optimizer, clip_grad)
        optimizer.step()

        loss_over_epoch += loss.data

        if i % 100 == 0 or i == len(train_dataloader) or i == 0:
            log_step(loss1.data, loss2.data, loss.data, i, len(train_dataloader), cur_epoch)

    return loss_over_epoch / len(train_dataloader)


def make_checkpoints_dir(save_path):

    os.mkdir(save_path + datetime.now().strftime('%Y%m%d'))


def main(args_):
    logging.info('Start train preparing')

    model_name = args_.model
    epoch_num = args_.epoch
    clip_grad = args_.clip
    save_path = args_.save
    lr = args_.lr

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = MODELS.get(model_name)().to(device)
    loss = LOSSES.get(args_.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    train_dataloader = data.get_loader(args_.rgb_path, args_.gt_path, args_.depth_path,
                                       batchsize=args_.batch_size, trainsize=args_.input_size, num_workers=2)
    val_dataset = data.test_dataset(
        args_.val_rgb_path, args_.val_gt_path, args_.val_depth_path, args_.input_size)

    make_checkpoints_dir(save_path)

    wandb_config = init_wnb_config(args_)

    with wandb.init(project=wandb_config['wandb_project'], config=wandb_config):
        # wandb.watch(model, loss, log="all")
        logging.info('Start training...')

        best_mae = 0.0
        best_epoch = 0

        for epoch in range(epoch_num):

            # Update learning rate
            cur_lr = bbs_utils.adjust_lr(
                optimizer, lr, epoch, args_.decay_rate, args_.decay_epoch)  # TODO change to scheduler?
            wandb.log({'lr': cur_lr})

            # Training epoch
            ave_loss = train_epoch(
                model, epoch, optimizer, train_dataloader, device, loss, clip_grad)
            logging.info(
                f'Epoch {epoch}/{epoch_num}, Average loss: {ave_loss:.4f}')
            wandb.log({'avg-loss': ave_loss, 'epoch': epoch})

            # Save model state
            if (epoch) % 10 == 0:
                logging.info(f'Save model state on {epoch} epoch')
                torch.save(model.state_dict(), save_path +
                           f'{model_name}_epoch_{epoch}.pth')

            # Validate model
            val_mae = validate(model, val_dataset, device, epoch)
            best_mae, best_epoch = update_best_mae(
                val_mae, best_mae, best_epoch, model, save_path, model_name, epoch)


if __name__ == '__main__':

    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S')

    parser = argparse.ArgumentParser(description='BBS models train script')

    parser.add_argument('--epoch', type=int, default=200, help='Epoch number')
    parser.add_argument('--model', type=str,
                        default='BBS-Net', help='Model name to train')
    parser.add_argument('--loss', type=str,
                        default='cross-entropy', help='Loss funtction')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int,
                        default=10, help='Batch size')
    parser.add_argument('--input-size', type=int,
                        default=352, help='Image size')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='Gradient clipping margin')
    parser.add_argument('--decay-rate', type=float,
                        default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--decay-epoch', type=int, default=60,
                        help='Every n epochs decay learning rate')
    parser.add_argument('--rgb-path', type=str,
                        default='../BBS_dataset/RGBD_for_train/RGB/', help='Path to train RGB images')
    parser.add_argument('--depth-path', type=str,
                        default='../BBS_dataset/RGBD_for_train/depth/', help='Path to train depth images')
    parser.add_argument('--gt-path', type=str,
                        default='../BBS_dataset/RGBD_for_train/GT/', help='Path to ground truth images')
    parser.add_argument('--val-rgb-path', type=str,
                        default='../BBS_dataset/RGBD_for_train/RGB/', help='Path to validation RGB images')
    parser.add_argument('--val-depth-path', type=str,
                        default='../BBS_dataset/RGBD_for_train/depth/', help='Path to validation depth images')
    parser.add_argument('--val-gt-path', type=str, default='../BBS_dataset/RGBD_for_train/GT/',
                        help='Path to validation ground truth images')
    parser.add_argument('--save', type=str,
                        default='./BBSNet_cpts/', help='Path to save model to')
    parser.add_argument('--wandb-project', type=str,
                        default='BBS-Net Train', help='Name of the WandB project')

    args = parser.parse_args()
    main(args)
