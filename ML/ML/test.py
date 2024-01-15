import csv
import pickle

from matplotlib import pyplot as plt
from tramsformer import TimeSeriesForcasting
from utils import save_checkpoint
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


#读取csv文件
def ReadFile(url):
    res = []
    with open(url, 'r') as f:
        reader = csv.reader(f)
        res = list(reader)[1:]
    return res

#将数据集进行滑动窗口划分,num为需要预测的天数
def Partition(line, num):
    train_set = []
    train_label = []
    l = len(line)
    n = l - num - 96 + 1  #n为滑动窗口的数量
    for i in range(n):
        train_set.append(line[i:i+96])
        train_label.append(line[i+96:i+96+num])
    return train_set, train_label

def TurnFloat(train_set):
    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            for k in range(len(train_set[i][j])):
                train_set[i][j][k] = float(train_set[i][j][k])
    return train_set

def test():
    print('test')

full_set1 = ReadFile('C:/Users/ASUS/PycharmProjects/ML/ML/ETT-small/train_set.csv')
full_set2 = ReadFile('C:/Users/ASUS/PycharmProjects/ML/ML/ETT-small/test_set.csv')
full_set3 = ReadFile('C:/Users/ASUS/PycharmProjects/ML/ML/ETT-small/validation_set.csv')
train_x, train_y = Partition(full_set1,96)
test_x, test_y = Partition(full_set2,96)
val_x, val_y = Partition(full_set3,96)

train_x = torch.tensor(TurnFloat(train_x))
train_y = torch.tensor(TurnFloat(train_y))
test_x = torch.tensor(TurnFloat(test_x))
test_y = torch.tensor(TurnFloat(test_y))
val_x = torch.tensor(TurnFloat(val_x))
val_y = torch.tensor(TurnFloat(val_y))
train_x = train_x.permute(0,2,1)
train_y = train_y.permute(0,2,1)
test_x = test_x.permute(0,2,1)
test_y = test_y.permute(0,2,1)
val_x = val_x.permute(0,2,1)
val_y = val_y.permute(0,2,1)

batch_size = 32
# 创建DataLoader
train_data = TensorDataset(train_x, train_y)
test_data = TensorDataset(test_x, test_y)
val_data = TensorDataset(val_x, val_y)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取统计参数
# with open('min_max_params.pickle', 'rb') as f:
#     loaded_min_max_params = pickle.load(f)
#     min_vals, max_vals = loaded_min_max_params
#     min_vals = torch.tensor(min_vals.values.reshape(1, 1, -1)).cuda()
#     max_vals = torch.tensor(max_vals.values.reshape(1, 1, -1)).cuda()
# 反正则化函数
# def min_max_denormalize(normalized_df):
#     return normalized_df * (max_vals - min_vals) + min_vals
with open('C:/Users/ASUS/PycharmProjects/ML/norm_params.pickle', 'rb') as f:
    scalers = pickle.load(f)
    mean_ = torch.tensor(scalers.mean_).to('cuda:0')
    std_ = torch.tensor(scalers.scale_).to('cuda:0')

# 定义 Transformer 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, output_size)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, :, :])  # 只取最后一个时间步的输出
        return out

def calculate_mae_mse(predictions, ground_truth):
    # 计算 MAE 和 MSE
    mae = np.mean(np.abs(predictions - ground_truth), axis=(0, 1))
    mse = np.mean((predictions - ground_truth) ** 2, axis=(0, 1))

    return mae, mse

#model = LSTMModel(96,64,96)
model = TimeSeriesForcasting(96,96)
epochs = 5
mode = 'test'

# model
model.cuda()
model.load_state_dict(torch.load('C:/Users/ASUS/PycharmProjects/ML/ML/checkpoints/model_23_0.7574584240263159.pth'))

# train
# 初始化用于存储每次实验结果的列表
all_predictions = []  # 用于存储每次实验的预测结果
average_feature_mae = 0
average_feature_mse = 0
overall_average_mae = 0
overall_average_mse = 0
model.eval()
for epoch in range(epochs):
    test_predictions = torch.Tensor()  # 初始化为空的张量
    cumulative_feature_mae = np.zeros(7)
    cumulative_feature_mse = np.zeros(7)
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            #outputs = model(X_batch)
            outputs = model((X_batch, X_batch))
            outputs = outputs.permute(0,2,1)
            y_batch = y_batch.permute(0,2,1)
            #y_batch = min_max_denormalize(y_batch)
            #output = min_max_denormalize(output)
            if test_predictions.numel() == 0:
                test_predictions = outputs.detach().cpu()  # 如果是第一次迭代，直接赋值
            else:
                test_predictions = torch.cat((test_predictions, outputs.detach().cpu()), dim=0)  # 否则拼接张量
            pred = outputs.detach().cpu().numpy()
            true = y_batch.detach().cpu().numpy()

            # 计算每个特征的 MAE 和 MSE
            feature_mae, feature_mse = calculate_mae_mse(pred, true)
            cumulative_feature_mae += feature_mae
            cumulative_feature_mse += feature_mse
            # 计算所有批次的平均 MAE 和 MSE
        average_feature_mae += cumulative_feature_mae / len(test_loader)
        average_feature_mse += cumulative_feature_mse / len(test_loader)
        # 计算整体的平均 MAE 和 MSE
        overall_average_mae += np.mean(cumulative_feature_mae / len(test_loader))
        overall_average_mse += np.mean(cumulative_feature_mse / len(test_loader))
        # 将此次实验的预测结果添加到总列表中
        all_predictions.append(test_predictions)
average_feature_mae = average_feature_mae / 5.0
average_feature_mse = average_feature_mse / 5.0
overall_average_mae = overall_average_mae / 5.0
overall_average_mse = overall_average_mse / 5.0
# 计算标准差
# 假设我们关注的是每个特征在整个验证集上的标准差
feature_std = torch.stack(all_predictions)
feature_std = torch.std(feature_std, dim=0)
feature_std = torch.sum(feature_std, dim=1)
feature_std = torch.mean(feature_std, dim=0)

print('total_mae: {0}, total_mse: {1}'.format(overall_average_mae, overall_average_mse))
print('mae: {0}, mse: {1}'.format(average_feature_mae, average_feature_mse))
print('feature_std: {}'.format(feature_std))
print('average_std: {}'.format(np.average(feature_std)))

