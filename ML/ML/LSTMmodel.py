from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, output_size)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, :, :])  # 只取最后一个时间步的输出
        return out