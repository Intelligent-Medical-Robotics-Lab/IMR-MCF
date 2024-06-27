import torch
from torch.utils.data.distributed import DistributedSampler
import pdb


def get_dataloader(dataset, batch_size: int, shuffle: bool):
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader



