import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from load_data import load_dataset

list_mat = ['B0005', 'B0006', 'B0007', 'B0018']
data_capacity, data_charge, data_discharge = load_dataset()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def build_sequences(list_data:list, window_size:int):
    """构建训练集和测试集时需要的裁剪序列"""
    x, y = [],[]
    for i in range(len(list_data) - window_size): # 每隔window_size分割数据
        sequence = list_data[i:i+window_size]
        target = list_data[i+1:i+1+window_size]
        x.append(sequence)
        y.append(target)
        
    return np.array(x), np.array(y) # 返回的是numpy数组

def get_train_test(data, name, window_size=8):
    """获取训练集和测试集"""
    data_sequence=data[name][1]
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_sequences(train_data, window_size)
    for k, v in data.items():
        if k != name:
            data_x, data_y = build_sequences(v[1], window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)

def relative_error(y_test, y_predict, threshold):
    """计算相对误差"""
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re

def evaluation(y_test, y_predict):
    """计算误差来评估模型"""
    mae = mean_absolute_error(y_test, y_predict) # 平均绝对误差
    mse = mean_squared_error(y_test, y_predict) # 均方误差
    rmse = sqrt(mean_squared_error(y_test, y_predict)) # 均方根误差
    return mae, rmse
    
def setup_seed(seed):
    """设置种子，使得实验可复现"""
    np.random.seed(seed)  # 设定numpy的随机种子
    random.seed(seed)  # 设定python的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed) # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False # 关闭cudnn的自动优化
        torch.backends.cudnn.deterministic = True # 保证每次结果一样

class Net(nn.Module):
    """定义网络结构"""
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, mode:str, n_class=1):
        super(Net, self).__init__()
        assert mode in ['LSTM', 'GRU', 'RNN'], 'mode should be LSTM, GRU or RNN'
        self.hidden_size = hidden_size

        if mode == 'LSTM':
            self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, n_class)
 
    def forward(self, x): 
        """前向传播"""
        out, _ = self.cell(x) 
        out = out.reshape(-1, self.hidden_size)
        out = self.linear(out) 
        return out

def train(lr:float, input_size:int, hidden_size:int, num_layers:int, weight_decay:float, mode:str, EPOCH:int, seed:int, rated_capacity:float):
    
    score_list, result_list = [], []
    for i in range(4):
        name = list_mat[i]
        train_x, train_y, train_data, test_data = get_train_test(data_capacity, name, window_size=input_size)
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, mode='LSTM')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_ = [0], []
        mae, rmse, re = 1, 1, 1
        score_, score = 1,1
        for epoch in range(EPOCH):
            X = np.reshape(train_x/rated_capacity,(-1, 1, input_size)).astype(np.float32)#(batch_size, seq_len, input_size)
            y = np.reshape(train_y[:,-1]/rated_capacity,(-1,1)).astype(np.float32)# shape 为 (batch_size, 1)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output= model(X)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            if (epoch + 1)%100 == 0:
                test_x = train_data.copy()    #每100次重新预测一次
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-input_size:])/rated_capacity,(-1, 1, input_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)                 # shape: (batch_size, 1, input_size)
                    pred = model(x)
                    next_point = pred.data.numpy()[0,0] * rated_capacity
                    test_x.append(next_point)                          #测试值加入原来序列用来继续预测下一个点
                    point_list.append(next_point)                     #保存输出序列最后一个点的预测值
                y_.append(point_list)                                 #保存本次预测所有的预测值
                loss_list.append(loss)
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=rated_capacity*0.7)
                print('epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae, rmse, re))
            score = [re, mae, rmse]
            if (loss < 1e-3) and (score_[0] < score[0]):
                break
            score_ = score.copy()
            
        score_list.append(score_)
        result_list.append(y_[-1])
    return score_list, result_list

def main():

    window_size = 16
    EPOCH = 1000
    lr = 0.001 # learning rate
    hidden_size = 128
    num_layers = 2
    weight_decay = 0.0
    mode = 'LSTM'        # RNN, LSTM, GRU
    Rated_Capacity = 2.0

    SCORE = []
    for seed in range(10):
        print('seed: ', seed)
        score_list, _ = train(lr=lr, input_size=window_size, hidden_size=hidden_size, num_layers=num_layers, 
                            weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed, rated_capacity=Rated_Capacity)
        print('------------------------------------------------------------------')
        for s in score_list:
            SCORE.append(s)

    mlist = ['re', 'mae', 'rmse']
    for i in range(3):
        s = [line[i] for line in SCORE]
        print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')

if __name__ == '__main__':
    main()