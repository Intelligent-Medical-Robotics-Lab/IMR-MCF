import torch
import torch.nn as nn
from utils.cal import generate_square_subsequent_mask
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, ResNet18_Weights, ResNet34_Weights
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP


class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=3, num_layers=10, device=None):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).cuda()
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).cuda()
        # import pdb;pdb.set_trace()
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out)
        return predictions
    

class MyResNet(nn.Module):
    def __init__(self, version='resnet18', num_classes=1000, angle_idx=None):
        super(MyResNet, self).__init__()
        self.version = version
        self.num_classes = num_classes
        self.base_model = None
        if version == 'resnet18':
            self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif version == 'resnet34':
            self.base_model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif version == 'resnet50':
            self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif version == 'resnet101':
            self.base_model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif version == 'resnet152':
            self.base_model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError('Invalid ResNet version')
        if angle_idx == 0:
            self.linear1 = nn.Linear(5, 33 * 33 * 3)
        elif angle_idx == 1 or angle_idx == 2 or angle_idx == 3:
            self.linear1 = nn.Linear(4, 33 * 33 * 3)
        self.linear2 = nn.Linear(1000, 3)
        

    def forward(self, x):
        x = self.linear1(x)
        x = torch.reshape(x, (x.shape[0], 3, 33, 33))
        x = self.base_model(x)
        x = self.linear2(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=256, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear4 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear5 = nn.Linear(hidden_layer_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    
    def forward(self, input_seq):
        x = self.linear1(input_seq)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        return x
    

class MyTransformer(nn.Module):
    def __init__(self, input_size=5, num_layers=6):
        super().__init__()
        self.transformer = nn.Transformer(d_model=32, batch_first=True, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=2048, nhead=8)
        self.linear_1 = nn.Linear(input_size, 32)
        self.linear_2 = nn.Linear(3, 32)
        self.linear_3 = nn.Linear(32, 3)
        
    def forward(self, inputs, targets):
        targets = self.linear_2(targets)
        inputs = self.linear_1(inputs)
        mask = generate_square_subsequent_mask(inputs.size(1)).cuda()
        output = self.transformer(inputs, targets, tgt_mask=mask)
        output = self.linear_3(output)
        return output
    

def load_model(model_name='', path='', angle_idx=None, load_weights=False, num_layers=None):
    model = None
    if model_name.lower() == 'lstm':
        if angle_idx == 0:
            model = LSTM(input_size=5, num_layers=num_layers)
        elif angle_idx == 1 or angle_idx == 2 or angle_idx == 3:
            model = LSTM(input_size=4, num_layers=num_layers)
    elif model_name.lower() == 'transformer':
        if angle_idx == 0:
            model = MyTransformer(input_size=5, num_layers=num_layers)
        elif angle_idx == 1 or angle_idx == 2 or angle_idx == 3:
            model = MyTransformer(input_size=4, num_layers=num_layers)
    else:
        raise BaseException
    if load_weights:
        state_dict = torch.load(path)
        unwarpped_state_dict = OrderedDict()
        # import pdb; pdb.set_trace()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            unwarpped_state_dict[name] = v
        model.load_state_dict(unwarpped_state_dict)
    return model
        


if __name__ == '__main__':
    model = load_model(model_name='resnet18', 
                     path='/mnt/data0/bingkui/IMR-MCF/ckpt_0/ResNet/resnet18_2024-03-12-23-09/best_model.pth',
                     angle_idx=0, 
                     load_weights=True)
