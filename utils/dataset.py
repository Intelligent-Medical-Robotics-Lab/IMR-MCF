from torch.utils.data import Dataset
from utils.process_file_utils import *
from tqdm import tqdm
import torch


class Dataset4LSTM(Dataset):
    def __init__(self, angle_path, jrf_path, patients_info_path, save, save_path='/fashuxu/bingkui/dataset/angle_{}.pt', angle_idx=0):
        super().__init__()
        self.angle_path = angle_path
        self.jrf_path = jrf_path
        self.patients_info_path = patients_info_path
        self.patients_weight = read_patient_info(self.patients_info_path)
        self.angle_files = list_files_in_directory(self.angle_path)
        self.jrf_files = list_files_in_directory(self.jrf_path)
        self.config = read_config('configs/data_config.yaml')
        self.features = None
        self.results = None
        self.angle_idx = angle_idx
        self.save_path = save_path.format(self.angle_idx)
        if save:
            self.process_data()
            self.save(self.save_path)
        
        
    def __getitem__(self, index):
        return self.features[index], self.results[index]
    
    
    def __len__(self):
        return len(self.features)
    
        
    def process_data(self):
        for file in tqdm(self.angle_files):
            name, direction = extract_name_direction(file)
            if direction == 'R':
                angle_columns = self.config['columns'][f'angle_{self.angle_idx}']['angle_right']
            elif direction == 'L':
                angle_columns = self.config['columns'][f'angle_{self.angle_idx}']['angle_left']
            jrf_columns = self.config['columns']['jrf']
            
            angle_data = read_rows_from_excel(os.path.join(self.angle_path, file), columns=angle_columns)
            jrf_data = read_rows_from_excel(os.path.join(self.jrf_path, file.replace('angle', 'JRF')), columns=jrf_columns)
            weight = self.patients_weight[name] * 9.8
            
            # 转换成张量
            angle_data_tensor = torch.tensor(angle_data)
            jrf_data_tensor = torch.tensor(jrf_data) / weight
            
            # 组装数据
            if self.features is None:
                self.features = angle_data_tensor.unsqueeze(0)
            else:
                self.features = torch.cat((self.features, angle_data_tensor.unsqueeze(0)), dim=0)
                
            if self.results is None:
                self.results = jrf_data_tensor.unsqueeze(0)
            else:    
                self.results = torch.cat((self.results, jrf_data_tensor.unsqueeze(0)), dim=0)
            
        
    def save(self, path):
        torch.save(self, path)
        
    @classmethod
    def load(cls, filename):
        return torch.load(filename)
            
            
class Dataset4ResNet(Dataset):
    def __init__(self, angle_path, jrf_path, patients_info_path, save, save_path='/mnt/data0/bingkui/IMR-MCF/dataset/Dataset4ResNet/Dataset4ResNet_angle_{}.pt', angle_idx=0):
        super().__init__()
        self.angle_path = angle_path
        self.jrf_path = jrf_path
        self.patients_info_path = patients_info_path
        self.patients_weight = read_patient_info(self.patients_info_path)
        self.angle_files = list_files_in_directory(self.angle_path)
        self.jrf_files = list_files_in_directory(self.jrf_path)
        self.config = read_config('configs/data_config.yaml')
        self.features = None
        self.results = None
        self.angle_idx = angle_idx
        self.save_path = save_path.format(self.angle_idx)
        if save:
            self.process_data()
            self.save(self.save_path)
        
        
    def __getitem__(self, index):
        return self.features[index], self.results[index]
    
    
    def __len__(self):
        return len(self.features)
    
        
    def process_data(self):
        for file in tqdm(self.angle_files):
            name, direction = extract_name_direction(file)
            if direction == 'R':
                angle_columns = self.config['columns'][f'angle_{self.angle_idx}']['angle_right']
            elif direction == 'L':
                angle_columns = self.config['columns'][f'angle_{self.angle_idx}']['angle_left']
            jrf_columns = self.config['columns']['jrf']
            
            angle_data = read_rows_from_excel(os.path.join(self.angle_path, file), columns=angle_columns)
            jrf_data = read_rows_from_excel(os.path.join(self.jrf_path, file.replace('angle', 'JRF')), columns=jrf_columns)
            weight = self.patients_weight[name] * 9.8
            
            # 转换成张量
            angle_data_tensor = torch.tensor(angle_data)
            jrf_data_tensor = torch.tensor(jrf_data) / weight
            
            # 组装数据
            if self.features is None:
                self.features = angle_data_tensor
            else:
                self.features = torch.cat((self.features, angle_data_tensor), dim=0)
                
            if self.results is None:
                self.results = jrf_data_tensor
            else:    
                self.results = torch.cat((self.results, jrf_data_tensor), dim=0)
            
        
    def save(self, path):
        torch.save(self, path)
        
    @classmethod
    def load(cls, filename):
        return torch.load(filename)
            
            
        
if __name__ == '__main__':
    pass
    