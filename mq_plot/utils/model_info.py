import timm
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def count_layers(model):
    """
    计算PyTorch模型的层数
    """
    return sum(isinstance(m, nn.Module) for m in model.modules())



def print_model(model):
    """
    打印模型的每一层的结构及其参数
    """
    print("Model structure:")
    print("----------------------------------------------------------------")
    for i, layer in enumerate(model.children()):
        print(f"Layer {i}: {layer}")
        print("----------------------------------------------------------------")
        for name, param in layer.named_parameters():
            print(f"\t{name}\t\t{param.shape}")
        print("----------------------------------------------------------------")

def print_model_demo():
    # 示例用法
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(512 * 4 * 4, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    num_layers = count_layers(model)
    print(f"模型层数：{num_layers}")
    print_model(model)

if __name__ == '__main__':
    model_name = 'vit_base_patch16_224'
    model = timm.create_model(model_name, pretrained=True)
    num_layers = count_layers(model)
    print(f"模型层数：{num_layers}")
    print_model(model)

