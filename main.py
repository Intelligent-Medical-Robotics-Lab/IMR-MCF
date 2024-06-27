import argparse
from utils.logger import build_logger
import random
import numpy as np
import torch
import torch.nn as nn
from utils.process_file_utils import read_config
import sys
import time
import os
from utils.dataset import Dataset4LSTM
from utils.train import train_one_epoch, valid_one_epoch
from torch.utils.data import random_split, DataLoader
from utils.models import load_model
from tqdm import tqdm
from utils.ddp_utils import get_dataloader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

def main(args, config):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = args.world_size
    setup(local_rank, world_size, config)
    
    # Logging and directory setup
    save_folder_path = args.save_folder_path
    training_time = time.strftime('%Y-%m-%d-%H-%M')
    if args.model_name == 'LSTM':
        training_time = f'{args.model_name}_{args.num_layers}_{training_time}' 
    elif args.model_name == 'Transformer':
        training_time = f'{args.model_name}_{args.num_layers}_{training_time}'
    else:
        logger.error(f"Undefined model: {args.model_name}")
        exit()
    save_folder_path = save_folder_path.format(args.angle_idx, args.model_name, training_time)
    if not os.path.exists(save_folder_path) and dist.get_rank() == 0:
        os.makedirs(save_folder_path)

    logger = build_logger('train', path=save_folder_path, rank=local_rank)
    if local_rank == 0:
        writer = SummaryWriter(log_dir=save_folder_path, comment=f'_{args.model_name}_angle_{args.angle_idx}')
        print(f'Create folder to save model ckpt: {save_folder_path}')
    logger.info(f'Create folder to save model ckpt: {save_folder_path}')

    # DDP setup
    model = load_model(model_name=args.model_name, angle_idx=args.angle_idx, num_layers=args.num_layers)
    device = torch.device("cuda", local_rank)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset_path = args.dataset_path.format(args.angle_idx)
    dataset = Dataset4LSTM.load(filename=dataset_path)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = get_dataloader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = get_dataloader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f'Split the dataset -> train_size: {train_size}, valid_size: {valid_size}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    min_loss = sys.maxsize
    for epoch_idx in tqdm(range(args.epoch)):
        train_loss = train_one_epoch(model=model,
                                      dataloader=train_dataloader, 
                                      optimizer=optimizer,
                                      criterion=criterion, 
                                      epoch=epoch_idx, 
                                      model_name=args.model_name, 
                                      logger=logger,
                                      device=device)
        valid_loss = valid_one_epoch(model=model, 
                                     dataloader=valid_dataloader, 
                                     criterion=criterion, 
                                     epoch=epoch_idx, 
                                     model_name=args.model_name, 
                                     logger=logger,
                                     device=device)
        if local_rank == 0:
            writer.add_scalar('Loss/train', train_loss, epoch_idx)
            writer.add_scalar('Loss/valid', valid_loss, epoch_idx)
        
        if (epoch_idx + 1) % 10 == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), f'{save_folder_path}/epoch_{epoch_idx + 1}.pth')
        
        if valid_loss < min_loss:
            min_loss = valid_loss
            if dist.get_rank() == 0:
                torch.save(model.state_dict(), f'{save_folder_path}/best_model.pth')
                logger.info(f'|SAVE| epoch: {epoch_idx + 1}, save the best model')
    if local_rank == 0:     
        writer.close()
    cleanup()

def setup(rank, world_size, config):
    random_seed = config['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    config = read_config('config/config.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=config['epoch'])
    parser.add_argument("--save_folder_path", type=str, default=config['save_folder_path'])
    parser.add_argument("--batch_size", type=int, default=config['batch_size'])
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=config['learning_rate'])
    parser.add_argument("--dataset_path", type=str, default='/fashuxu/bingkui/dataset/angle_{}.pt')
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--angle_idx", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    main(args, config)
