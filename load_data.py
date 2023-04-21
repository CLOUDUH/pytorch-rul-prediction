import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime

def convert_time(hmm): 
    """把matlab的时间转换成python的时间"""
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

def read_mat(matfile:str):
    """加载数据集mat文件"""
    data_mat = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]

    col = data_mat[filename][0][0][0][0]
    size = col.shape[0]

    dataset = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_time(col[i][2][0])), d2
        dataset.append(d1)
    return dataset

def get_cycle(dataset:dict):
    """获取电池的容量数据和循环数据"""
    cycle, capacity = [], []
    i = 1
    for battery in dataset:
        if battery['type'] == 'discharge':
            capacity.append(battery['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

def get_value(dataset:dict, type:str):
    """获取锂电池充电或放电时的测试数据"""
    data=[]
    for battery in dataset:
        if battery['type'] == type:
            data.append(battery['data'])
    return data

def load_dataset():
    """加载数据集"""

    list_mat = ['B0005', 'B0006', 'B0007', 'B0018']
    dir_path = 'dataset/'

    data_capacity = {}
    data_charge = {}
    data_discharge = {}

    for name in list_mat:
        print('Loading dataset ' + name + '.mat ...')
        path = dir_path + name + '.mat'
        data = read_mat(path)
        data_capacity[name] = get_cycle(data) # 循环容量数据
        data_charge[name] = get_value(data, 'charge') # 充电数据
        data_discharge[name] = get_value(data, 'discharge') # 放电数据

    return data_capacity, data_charge, data_discharge