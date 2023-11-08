import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torchviz import make_dot

# 导入pyplot库
import matplotlib.pyplot as plt

# 定义一个函数，绘制模型结构图
def plot_network(model, input_size):
    """绘制网络结构图"""
    x = torch.randn(*input_size)
    # 将输入数据转换为4D的，即(1, 3, 224, 224)
    x = x.unsqueeze(0)

    y = model(x)
    g = make_dot(y, params=dict(model.named_parameters()))
    g.format = 'png'
    g.render('model', view=False)

    # 读取并显示绘制好的模型结构图
    img = plt.imread('model.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 使用timm中的模型
    model = timm.create_model('resnet18', pretrained=False)
    input_size = (3, 224, 224)
    plot_network(model, input_size)

    # 使用torchvision中的模型
    model = models.resnet18(pretrained=False)
    input_size = (3, 224, 224)
    plot_network(model, input_size)