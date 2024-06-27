import torch


def calculate_rmse(output, target):
    assert output.shape == target.shape, "Output and target must have the same shape"
    
    mse = torch.mean((output - target) ** 2)
    
    rmse = torch.sqrt(mse)
    
    return rmse.item()


def calculate_ae(outputs, targets):
    """
    计算绝对误差
    """
    return torch.mean(torch.abs(outputs - targets)).item()


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == '__main__':
    a = [1., 2., 3.]
    b = [1., 2., 3.]
    print(calculate_ae(a, b))
