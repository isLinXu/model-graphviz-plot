import timm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_network(model, input_size = (1, 28, 28)):
    """绘制网络结构图"""
    # input_size = (1, 28, 28)
    x = torch.randn(1, *input_size)
    out = model(x)

    plt.figure(figsize=(10, 10))
    plt.title("Network Architecture")

    # 绘制输入层
    plt.subplot(1, len(model)+1, 1)
    plt.axis('off')
    plt.imshow(x[0][0].detach().numpy(), cmap='gray')
    plt.title('Input\n{}'.format(input_size))

    # 绘制隐藏层和输出层
    for i, layer in enumerate(model):
        plt.subplot(1, len(model)+1, i+2)
        plt.axis('off')
        plt.imshow(out[0][i].detach().numpy(), cmap='gray')
        plt.title('{}\n{}'.format(layer.__class__.__name__, list(out.shape)[1:]))

model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

input_size =  (1, 28, 28)

# model_name = 'vit_base_patch16_224'
# model = timm.create_model(model_name, pretrained=True)

plot_network(model, input_size = (input_size))
plt.show()