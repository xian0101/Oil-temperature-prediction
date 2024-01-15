import csv
import pickle

import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from LSTMmodel import LSTMModel
from tramsformer import TimeSeriesForcasting

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


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = LSTMModel(96, 64, 96)
model = TimeSeriesForcasting(96,96)
model.load_state_dict(torch.load('C:/Users/ASUS/PycharmProjects/ML/ML/checkpoints/model_23_0.7574584240263159.pth'))

model.to(device)
model.eval()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

with open('C:/Users/ASUS/PycharmProjects/ML/norm_params.pickle', 'rb') as f:
    scalers = pickle.load(f)
    mean_ = torch.tensor(scalers.mean_).resize(1, 1, 7).to('cuda:0')
    std_ = torch.tensor(scalers.scale_).resize(1, 1, 7).to('cuda:0')

X_test_batch, y_test_batch = test_x[0], test_y[0]
X_test_batch = X_test_batch.unsqueeze(1)
y_test_batch = y_test_batch.unsqueeze(1)
X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
#test_output = model((X_test_batch))
test_output = model((X_test_batch, X_test_batch))

# X_test_batch = X_test_batch.permute(1,2,0)
# test_output = test_output.permute(1,2,0)
# y_test_batch = y_test_batch.permute(1,2,0)
# X_test_batch = X_test_batch.squeeze()
# test_output = test_output.squeeze()
# y_test_batch = y_test_batch.squeeze()
# X_test_batch = X_test_batch.tolist()
# y_test_batch = y_test_batch.tolist()
# test_output = test_output.tolist()
# output = torch.permute(test_output, (0, 2, 1))
test_output = test_output.permute(1,2,0)
y_test_batch = y_test_batch.permute(1,2,0)
X_test_batch = X_test_batch.permute(1,2,0)
print(test_output.shape)
print(std_.shape)
test_output = (test_output * std_) + mean_
y_test_batch = (y_test_batch * std_) + mean_
X_test_batch = (X_test_batch * std_) + mean_
print(X_test_batch.shape, y_test_batch.shape)
X = torch.cat((X_test_batch.detach().cpu(), y_test_batch.detach().cpu()), dim=1)
X = X.permute(2, 1, 0).reshape(7, 96+96)
outputs = test_output.detach().cpu().permute(2, 1, 0).reshape(7, 96)
plt.figure(figsize=(10, 6))
plt.plot([i for i in range(0, 192)], X[6], label='GroundTruth', color='orange')
plt.plot([i for i in range(96, 192)], outputs[6], label='Prediction', color='blue')
plt.legend()
plt.title('Oil Temperature')
plt.xlabel('Time Steps')
plt.ylabel('Oil Temperature')
plt.savefig('comparision.png')


# X_test_batch, y_test_batch = test_x[0], test_y[0]
# X_test_batch = X_test_batch.unsqueeze(1)
# y_test_batch = y_test_batch.unsqueeze(1)
# X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
# test_output = model(X_test_batch)
# test_output = test_output.permute(1,2,0)
# y_test_batch = y_test_batch.permute(1,2,0)
# test_output = test_output.squeeze()
# y_test_batch = y_test_batch.squeeze()
# y_test_batch = y_test_batch.tolist()
# test_output = test_output.tolist()
#
# # 创建包含7张子图的图表
# fig, axes = plt.subplots(1, 7, figsize=(14, 4))
#
# # 创建包含7张子图的图表
# fig, axes = plt.subplots(1, 7, figsize=(14, 4))
#
# # 遍历每个子图，并在其上绘制数据
# for i in range(7):
#     axes[i].plot(test_output[i], label='prediction')
#     axes[i].plot(y_test_batch[i], label='truth')
#     axes[i].set_title(f'Subplot {i + 1}')
#     axes[i].legend()
#
# # 调整子图布局
# plt.tight_layout()
#
# # 显示图表
# plt.show()