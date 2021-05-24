import argparse
import logging
import sys
import torch
import numpy as np

from model_zoo.BBSNet.models.BBSNet_model import BBSNet
from model_zoo.BBSNet.models.BBSNet_model_effnet import BBSNet as BBSNet_effnet


MODELS = {
    'BBS-Net': BBSNet(),
    'BBS-Net-Effnet': BBSNet_effnet()
}


def prepare_model(model_name, model_path, device):
    model = MODELS.get(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)


def main(args_):
    '''
        Script to Measure Inference Time of BBS-Net
        Reference: https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    '''
    device = torch.device('cuda:0')
    model = prepare_model(args_.model, args_.model_path, device)
    inp_size = args_.input_size
    channel_num = args_.channel_num

    dummy_input = torch.randn(1, channel_num, inp_size,
                              inp_size, dtype=torch.float).to(device)

    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    logging.info(f'Measuring performance for {args_.model}({args_.model_path}). Input size: {inp_size}. Number of channels: {channel_num}')
    # GPU-WARM-UP
    logging.info('Start GPU warming up')
    for _ in range(10):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    logging.info('Start performance measuring')
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    logging.info('Measuring finished')
    logging.info(f'Mean inference time: {mean_syn}Std inference time: {std_syn}\n')


if __name__ == '__main__':

    logging.basicConfig(
        stream=sys.stdout,  format='[%(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %I:%M:%S')

    parser = argparse.ArgumentParser(description='BBS models train script')

    parser.add_argument('--input-size', type=int,
                        default=352, help='Image size')
    parser.add_argument('--channel', type=int,
                        default=3, help='Channel number')
    parser.add_argument('--model', type=str,
                        default='BBS-Net', help='Model name')
    parser.add_argument('--model-path', type=str,
                        default='', help='Path to model to evaluate')

    args = parser.parse_args()
    main(args)
