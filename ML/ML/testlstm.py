import torch
import torch.nn as nn

class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, :, :])  # 只取最后一个时间步的输出
        return out

# 创建新的输入数据（假设batch_size=10，sequence_length=3，input_size=4）
new_input_data = torch.randn(10, 3, 4)

# 修改模型
model = CustomLSTMModel(input_size=4, hidden_size=32, output_size=4)

# 前向传播
new_output = model(new_input_data)

# 输出的维度
print(new_output.shape)  # 应该为 [10, your_output_size]
