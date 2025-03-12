import torch


class NLSAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.fc2 = torch.nn.Linear(input_dim // 2, input_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        z = torch.mean(x, dim=0, keepdim=True)  # [1, D]
        # 全连接层压缩和恢复维度
        w = self.fc1(z)
        w = self.relu(w)
        w = self.fc2(w)
        # 生成注意力权重
        s = self.sigmoid(w)
        # 应用注意力
        x_hat = x * s
        return x_hat