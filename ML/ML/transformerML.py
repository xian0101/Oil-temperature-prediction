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
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, dim_feedforward, output_size):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_size, nhead=num_heads,
                                         num_encoder_layers=num_layers,
                                         num_decoder_layers=num_layers,
                                         dim_feedforward=dim_feedforward,
                                         batch_first=True)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return self.linear(out[:, :, :])
model = TimeSeriesForcasting(
        n_encoder_inputs=int(train_x.shape[2]),
        n_decoder_inputs=int(train_x.shape[2]),
        lr=1e-5,
        dropout=0.1,
    )
lr = 0.00001
epochs = 50
mode = 'train'

# 损失函数和优化器
criterion = nn.MSELoss()  # MAE损失
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses = [], []
if mode != 'train':
    test()
else:
    # model
    model.cuda()
    best_loss = 100.0
    # train
    for epoch in range(epochs):
        model.train()
        print('\nEpoch: [%d | %d]' % (epoch + 1, epochs))

        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            #output = model(X_batch)
            output = model((X_batch, X_batch))
            #y_batch = min_max_denormalize(y_batch)
            #output = min_max_denormalize(output)
            output = output.view(-1)
            y_batch = y_batch.view(-1)

            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                #val_output = model(X_val_batch)
                val_output = model((X_val_batch, X_val_batch))
                #y_val_batch = min_max_denormalize(y_val_batch)
                #val_output = min_max_denormalize(val_output)
                val_output = val_output.view(-1)
                y_val_batch = y_val_batch.view(-1)
                val_loss = criterion(val_output, y_val_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # save_model
        checkpoint = 'C:/Users/ASUS/PycharmProjects/ML/ML/checkpoints'
        is_best = avg_val_loss < best_loss
        best_loss = min(avg_val_loss, best_loss)
        save_checkpoint({
            'fold': 0,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, single=True, checkpoint=checkpoint)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./loss.png')
    plt.show()

checkpoint = torch.load('C:/Users/ASUS/PycharmProjects/ML/ML/checkpoints/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
X_test_batch, y_test_batch = test_x[0], test_y[0]
X_test_batch = X_test_batch.unsqueeze(1)
y_test_batch = y_test_batch.unsqueeze(1)
X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
test_output = model((X_test_batch, X_test_batch))
test_output = test_output.permute(1,2,0)
y_test_batch = y_test_batch.permute(1,2,0)
test_output = test_output.squeeze()
y_test_batch = y_test_batch.squeeze()
y_test_batch = y_test_batch.tolist()
test_output = test_output.tolist()

# 创建包含7张子图的图表
fig, axes = plt.subplots(1, 7, figsize=(14, 4))

# 创建包含7张子图的图表
fig, axes = plt.subplots(1, 7, figsize=(14, 4))

# 遍历每个子图，并在其上绘制数据
for i in range(7):
    axes[i].plot(test_output[i], label='prediction')
    axes[i].plot(y_test_batch[i], label='truth')
    axes[i].set_title(f'Subplot {i + 1}')
    axes[i].legend()

# 调整子图布局
plt.tight_layout()

# 显示图表
plt.show()