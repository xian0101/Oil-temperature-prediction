import torch
import torch.nn as nn
import torch.optim as optim
from tramsformer import TimeSeriesForcasting
from prepare_data import get_dataloader
from args import args
from torch.utils.data import Dataset, DataLoader
from prepare_data import TimeSeriesDataset
from utils import save_checkpoint
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # 只取最后一个时间步的输出
        return out

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
        return self.linear(out[:, -1, :])

def test():
    print('test')

if __name__ == "__main__":  
    # 创建数据集
    X_train, y_train, X_val, y_val, X_test, y_test = get_dataloader(args.train_dataset_path, args.test_dataset_path, args.val_dataset_path)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = args.batch_size
	
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型
    # lstm
    # input_size = X_train.shape[2]  # 特征数量
    # hidden_size = 50  # LSTM单元数
    # output_size = y_train.shape[1]  # 输出步数
    # model = LSTMModel(input_size, hidden_size, output_size).to(device)
    # transformer
    model = TimeSeriesForcasting(
        n_encoder_inputs=int(X_train.shape[2]),
        n_decoder_inputs=int(X_train.shape[2]),
        lr=1e-5,
        dropout=0.1,
    )


    # 损失函数和优化器
    criterion = nn.L1Loss()  # MAE损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    if args.mode != 'train':
        test()
    else:
        # model
        model.cuda()
        best_loss = 100.0
        # train
        for epoch in range(args.epochs):
            model.train()
            print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                # output = model(X_batch)
                
                output = model((X_batch, X_batch))

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
                    # val_output = model(X_val_batch)
                    val_output = model((X_val_batch, X_val_batch))   

                    val_output = val_output.view(-1)
                    y_val_batch = y_val_batch.view(-1)

                    val_loss = criterion(val_output, y_val_batch)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # save_model
            is_best = avg_val_loss < best_loss
            best_loss = min(avg_val_loss, best_loss)
            save_checkpoint({
                        'fold': 0,
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'avg_train_loss':avg_train_loss,
                        'avg_val_loss': avg_val_loss,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, single=True, checkpoint=args.checkpoint)
            print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
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
