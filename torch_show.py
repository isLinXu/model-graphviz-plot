import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys


# 定义一个简单的卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型对象
model = Net()

# 重定向sys.stdout到文件
sys.stdout = open('model_summary.txt', 'w')
# 打印模型摘要信息
summary(model, (1, 28, 28))
# 关闭文件和恢复sys.stdout
sys.stdout.close()
sys.stdout = sys.__stdout__

# 读取模型摘要信息文件并打印
with open('model_summary.txt', 'r') as f:
    summary_text = f.read()
    print(summary_text)

