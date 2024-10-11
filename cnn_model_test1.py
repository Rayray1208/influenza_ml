import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv("merged_file.csv")
data['Year'] = data['YearWeek'].astype(str).str[:4].astype(int)
data['Week'] = data['YearWeek'].astype(str).str[4:].astype(int)

# 提取特征和目标
X = data[['Year', 'Week', 'ExcludedCases', 'PendingCases', "AverageTemperature"]]
y = data['ConfirmedCases']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # 将 y 转换为适合的形状

# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 64)  # 更大的第一层
        self.fc2 = nn.Linear(64, 32)  # 新增中间层
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # 中间层激活
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 模拟训练过程
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # 前向传播
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # 每10个 epoch 输出一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 训练完成后，可以绘制损失曲线
import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()