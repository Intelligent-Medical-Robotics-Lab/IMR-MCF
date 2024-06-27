import torch
from torch.utils.data import DataLoader, random_split
from utils.process_file_utils import read_config
from utils.dataset import Dataset4LSTM
from utils.models import load_model
from torch import nn
from tqdm import tqdm
import random
import numpy as np
from utils.cal import calculate_rmse, calculate_ae
import matplotlib.pyplot as plt
from utils.process_file_utils import save_dict, load_dict
import os
from pathlib import Path
import matplotlib.patches as mpatches
device_ids = [0]
file_format = ['png', 'pdf', 'svg']
force = ['KJCF', 'MCF', 'LCF']


def setup(config):
    """
    初始化
    """
    random_seed = config['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    

def get_total_ae(args, config):
    """
    Load model
    """
    models =  {'lstm_2': None, 
               'lstm_4': None,
               'lstm_6': None,
               'lstm_12': None,
               'lstm_18': None,
               'transformer_2': None,
               'transformer_4': None,
               'transformer_6': None}
    angle_idx = args.angle_idx
    dataset_path = args.dataset_path.format(angle_idx)
    dataset = Dataset4LSTM.load(filename=dataset_path)
    train_size = int(0.80 * len(dataset))
    valid_size = int(len(dataset) - train_size)
    _, valid_dataset = random_split(dataset, [train_size, valid_size])
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    for model_name in models.keys():
        tqdm.write(f"加载模型{model_name}中")
        num_layers = int(model_name.split('_')[1])
        name = model_name.split('_')[0]
        model = load_model(model_name=name, 
                           angle_idx=angle_idx, 
                           num_layers=num_layers,
                           load_weights=True,
                           path=config['bestmodel'][f'angle_{angle_idx}'][model_name]).cuda()
        models[model_name] = model
    """
    Calculate AE
    """
    result = {f'feature_{feature_idx}': {model_name: None for model_name in models.keys()} for feature_idx in range(3)}
    for feature_idx in range(3):
        for model_name, model in models.items():
            ae = .0
            for _, (inputs, targets) in tqdm(enumerate(valid_dataloader), desc=f"计算Model:{model_name}对于Feature{feature_idx + 1}的RMSE中"):
                inputs, targets = inputs.cuda(), targets.cuda()
                if model_name.startswith('lstm'):
                    outputs = model(inputs)
                elif model_name.startswith('transformer'):
                    outputs = model(inputs, targets)
                outputs = outputs[:, :, feature_idx]
                targets = targets[:, :, feature_idx]
                ae += calculate_ae(outputs, targets)
            result[f'feature_{feature_idx}'][model_name] = ae
    return result


def get_rmse(args, config):
    """
    Load model
    """
    models =  {'lstm_2': None, 
               'lstm_4': None,
               'lstm_6': None,
               'lstm_12': None,
               'lstm_18': None,
               'transformer_2': None,
               'transformer_4': None,
               'transformer_6': None}
    angle_idx = args.angle_idx
    dataset_path = args.dataset_path.format(angle_idx)
    dataset = Dataset4LSTM.load(filename=dataset_path)
    train_size = int(0.80 * len(dataset))
    valid_size = int(len(dataset) - train_size)
    _, valid_dataset = random_split(dataset, [train_size, valid_size])
    valid_dataloader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    for model_name in models.keys():
        tqdm.write(f"加载模型{model_name}中")
        num_layers = int(model_name.split('_')[1])
        name = model_name.split('_')[0]
        model = load_model(model_name=name, 
                           angle_idx=angle_idx, 
                           num_layers=num_layers,
                           load_weights=True,
                           path=config['bestmodel'][f'angle_{angle_idx}'][model_name]).cuda()
        models[model_name] = model
    """
    Calculate RMSE
    """
    result = {f'feature_{feature_idx}': {model_name: None for model_name in models.keys()} for feature_idx in range(3)}
    for feature_idx in range(3):
        for model_name, model in models.items():
            rmse = .0
            for _, (inputs, targets) in tqdm(enumerate(valid_dataloader), desc=f"计算Model:{model_name}对于Feature{feature_idx + 1}的RMSE中"):
                inputs, targets = inputs.cuda(), targets.cuda()
                if model_name.startswith('lstm'):
                    outputs = model(inputs)
                elif model_name.startswith('transformer'):
                    outputs = model(inputs, targets)
                outputs = outputs[:, :, feature_idx]
                targets = targets[:, :, feature_idx]
                rmse += calculate_rmse(outputs, targets)
            result[f'feature_{feature_idx}'][model_name] = rmse
    return result


def Total_AE(args, file, config):
    file = file.format(args.angle_idx)
    if not os.path.isfile(file):
        ae = get_total_ae(args, config=config)
        save_dict(ae, file)
        

def Total_RMSE(args, file, config):
    file = file.format(args.angle_idx)
    if not os.path.isfile(file):
        rmse = get_rmse(args, config=config)
        save_dict(rmse, file)
                     
if __name__ == '__main__':
    import argparse
    config = read_config('config/config.yaml')
    setup(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle_idx", type=int, default=3)
    parser.add_argument("--dataset_path", type=str, default='/fashuxu/bingkui/dataset/angle_{}.pt')
    args=parser.parse_args()
    for i in tqdm(range(4)):
        tqdm.write(f'计算angle_{i}中')
        args.angle_idx = i # this operation is not elegant honestly.
        Total_AE(args, config=config, file='/fashuxu/bingkui/statistic/total_AE/total_ae_angle_{}.json')
        Total_RMSE(args, config=config, file='/fashuxu/bingkui/statistic/RMSE/rmse_angle_{}.json')
