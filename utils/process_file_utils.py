import pandas as pd
import os
import yaml
import json


def read_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_config(filename, config):
    with open(filename, 'w') as file:
        yaml.safe_dump(config, file)


def read_patient_info(filename):
    df = pd.read_excel(filename)
    df['name'] = df['name'].astype(str)
    data_dict = df.set_index('name')['weight'].to_dict()
    return data_dict


def list_files_in_directory(directory):
    all_files_and_dirs = os.listdir(directory)
    files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(directory, f))]
    return files


def read_rows_from_excel(filename, columns):
    df = pd.read_excel(filename)
    selected_columns = df.iloc[:, columns]
    column_list = selected_columns.values.tolist()
    return column_list

def extract_name_direction(file):
    name, direction = file.split('_')[0], file.split('_')[-1].split('.')[0]
    if name[0].isalpha():
        name = name.split('0')[0]  
    return name, direction 


def save_dict(d, filename):
    """
    保存字典
    """
    with open(filename, 'w') as f:
        json.dump(d, f)


def load_dict(filename):
    """
    加载字典
    """
    with open(filename, 'r') as f:
        return json.load(f)



if __name__ == '__main__':
    config = read_config('D:\CODEING\Python\IMR-MCF\configs\data_config.yaml')
    columns = config['columns']['angle_left']
    print(read_rows_from_excel('data/angles/2014001_C3_01_angle_L.xlsx', columns))
    