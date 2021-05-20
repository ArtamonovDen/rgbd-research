import sys
import os
import logging
import argparse
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from metrics import metrics

from model_zoo.BBSNet.models.BBSNet_model import BBSNet
from model_zoo.BBSNet.models.BBSNet_model_effnet import BBSNet as BBSNet_effnet
from model_zoo.BBSNet import data

MODELS = {
    'BBS-Net': BBSNet(),
    'BBS-Net-Effnet': BBSNet_effnet()
}

# TEST_DATASETS = ['STERE', 'NJU2K_TEST',
#                  'NLPR_TEST', 'NLPR', 'DES', 'SSD', 'LFSD', 'DUT-RGBD', 'SIP']

TEST_DATASETS = ['NLPR']


def init_wnb_config(args_):
    return dict(
        wandb_project=args_.wandb_project,
        experiment_id=args_.experiment_id,
        model_name=args_.model,
        input_size=args_.input_size
    )


def prepare_model(model_name, model_path, device):

    model = MODELS.get(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def get_test_path(dataset_path, dataset):
    rgb_path = dataset_path + dataset + '/RGB/'
    gt_path = dataset_path + dataset + '/GT/'
    depth_path = dataset_path + dataset + '/depth/'
    return rgb_path, depth_path, gt_path


def main(args_):

    logging.info('Prepare evaluation parameters')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = prepare_model(args_.model, args_.model_path, device)
    dataset_path = args_.test_path
    input_size = args_.input_size

    config = init_wnb_config(args_)
    with wandb.init(project=config['wandb_project'], config=config):

        for dataset in TEST_DATASETS:
            logging.info(f'Processing {dataset}')

            if not os.path.exists(dataset_path + dataset):
                logging.info(
                    f'Dataset {dataset_path + dataset} does not exist')
                continue

            total_f1, total_mae = 0.0, 0.0

            rgb_path, depth_path, gt_path = get_test_path(
                dataset_path, dataset)
            test_loader = data.test_dataset(
                rgb_path, gt_path, depth_path, input_size)

            for i in range(test_loader.size):
                image, gt, depth, name, image_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depth = depth.cuda()
                _, res = model(image, depth)

                res = F.upsample(res, size=gt.shape,
                                 mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                mae = metrics.mae(res, gt)
                f1 = metrics.f_beta_measure(res, gt)

                wandb.log({f'mae_{dataset}': mae, f'f1{dataset}': f1})
                wandb.log({'res': [wandb.Image(torch.tensor(res))], 'gt': [
                          wandb.Image(torch.tensor(gt))]})
                total_mae += mae
                total_f1 += f1

            logging.info(f'[{dataset}] Average f1 is {total_f1 / test_loader.size:.3f}')
            logging.info(f'[{dataset}] Average mae is {total_mae / test_loader.size:.3f}')

            # TODO create by-experiment table
            # TODO add results to all experiments table


if __name__ == '__main__':

    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S')

    parser = argparse.ArgumentParser(description='BBS models train script')

    parser.add_argument('--input-size', type=int,
                        default=352, help='Image size')
    parser.add_argument('--test-path', type=str,
                        default='./datasets/', help='Path to test datasets')
    parser.add_argument('--model', type=str,
                        default='BBS-Net', help='Model name')
    parser.add_argument('--model-path', type=str,
                        default='', help='Path to model to evaluate')
    parser.add_argument('--experiment-id', type=str,
                        default='-1', help='Experiment id to save it with')
    parser.add_argument('--wandb-project', type=str,
                        default='BBS-Net Evaluation', help='Name of the WandB project')

    args = parser.parse_args()
    main(args)
