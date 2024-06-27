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
    
      
@torch.no_grad()  
def process_results_to_ae(args, config):
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
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    for model_name in models.keys():
        print(f"加载模型{model_name}中")
        num_layers = int(model_name.split('_')[1])
        name = model_name.split('_')[0]
        model = load_model(model_name=name, 
                           angle_idx=angle_idx, 
                           num_layers=num_layers,
                           load_weights=True,
                           path=config['bestmodel'][f'angle_{angle_idx}'][model_name]).cuda()
        models[model_name] = model

    result = {}
    for feature_idx in range(3):
        best_ae = float('inf')
        for _, (inputs, targets) in tqdm(enumerate(valid_dataloader), desc=f"计算Feature{feature_idx + 1}的AE中"):
            total_ae = .0
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_targets = {'ground_truth': targets[0][:, feature_idx].cpu().numpy().tolist()}
            for model_name, model in models.items():
                if model_name.startswith('lstm'):
                    outputs = model(inputs)[0][:, feature_idx]
                elif model_name.startswith('transformer'):
                    outputs = model(inputs, targets)[0][:, feature_idx]
                outputs_targets[model_name] = outputs.detach().cpu().numpy().tolist()
                ae = calculate_ae(outputs, targets[0][:, feature_idx])
                total_ae += ae
            if total_ae < best_ae:
                best_ae = total_ae
                result[feature_idx] = outputs_targets
    return result
    
        
@torch.no_grad()
def prediction_plot(args, config, file):
    file = file.format(args.angle_idx)
    if os.path.isfile(file):
        ae = load_dict(file)
    else:
        ae = process_results_to_ae(args, config=config)
        save_dict(ae, file)
        ae = load_dict(file) # 这个必须要，不然会有 int 和 str 的bug
    for feature_idx in range(3):
        output_path = f'/fashuxu/bingkui/figure/prediction_plots/angle_{args.angle_idx}'
        if not Path(output_path).exists():
            Path(output_path).mkdir(parents=True, exist_ok=True)
        plot_predictions(predictions=ae, path=output_path, feature_idx=feature_idx)
        
        
def plot_predictions(predictions, path, feature_idx):
    plt.figure(figsize=(10, 6))
    model_results = {model_name: prediction for model_name, prediction in predictions[str(feature_idx)].items()}
    for model_name, prediction in model_results.items():
        if model_name == 'ground_truth':
            plt.plot(prediction, label=model_name, marker='x')
        else:
            plt.plot(prediction, label=model_name, alpha=0.7)

    plt.ylabel('Value (N)')
    plt.title('Model Predictions vs Ground Truth')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True)
    file_format
    for format in file_format:
        plt.savefig(path + f'/feature_{feature_idx}.' + format, format=format)
    plt.close()

                     
if __name__ == '__main__':
    import argparse
    config = read_config('config/config.yaml')
    setup(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle_idx", type=int, default=3)
    parser.add_argument("--dataset_path", type=str, default='/fashuxu/bingkui/dataset/angle_{}.pt')
    args=parser.parse_args()
    prediction_plot(args, config=config, file='/fashuxu/bingkui/statistic/best_instance_AE/ae_angle_{}.json')
