import pickle
from multiprocessing.spawn import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import sys
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, continue_run, eval_run

if __name__ == "__main__":
    freeze_support()
    # config_file = Path(r"C:\Users\lucas\PycharmProjects\_specialkursus\us_camels_run_configs\config_US_LSTM_ATTRIBUTES.yaml")

    # display python version
    print(f'Python version: {torch.__version__}')
    # display CUDA version
    print(f'CUDA version: {torch.version.cuda}')
    # display CUDA device count and name
    print(f'CUDA device count: {torch.cuda.device_count()}')

    # Default CPU-only mode if cuda not available
    gpu = -1

    if torch.cuda.is_available():
        print(f'Available CUDA device: {torch.cuda.current_device()}')
        print(f'CUDA device name: {torch.cuda.get_device_name("cuda:0")}')
        gpu = 0

    # start_run(config_file=config_file, gpu=gpu) 
    # continue_run(run_dir=Path(r"D:\_specialkursus\code\runs\dk_ealstm_0510_182415"), gpu=gpu)
    eval_run(Path(r'C:\Users\lucas\PycharmProjects\_specialkursus\runs\nse_dk_lstm_attributes_per_basin_365_1810_112154'), period='test', gpu=gpu)
