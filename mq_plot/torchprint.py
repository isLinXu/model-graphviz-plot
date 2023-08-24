import timm
import torch
# from torchsummary import summary
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydotplus
from torchviz import make_dot
import sys
from torchinfo import summary

# # model = ConvNet()
# batch_size = 16
# summary(model, input_size=(batch_size, 1, 28, 28))

def model_info_print(model, input_size,model_name):
    # 重定向sys.stdout到文件
    sys.stdout = open(model_name + '_model_summary.txt', 'w')
    # 打印模型摘要信息
    try:
        # summary(model, input_size)
        summary(model, input_size=input_size)
    except Exception as e:
        print("Error: ", e)
    # 关闭文件和恢复sys.stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
        # 读取模型摘要信息文件并打印
        print("model_name:",model_name)
    # 读取模型摘要信息文件并打印
    with open(model_name+'_model_summary.txt', 'r') as f:
        summary_text = f.read()
        print(summary_text)


def model_summary_print(model,input_size,model_name):
    # 重定向sys.stdout到文件
    sys.stdout = open(model_name + '_model_summary.txt', 'w')
    # 打印模型摘要信息
    try:
        summary(model, input_size)
    except Exception as e:
        print("Error: ", e)
    # 关闭文件和恢复sys.stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # 读取模型摘要信息文件并打印
    with open(model_name+'_model_summary.txt', 'r') as f:
        summary_text = f.read()
        print(summary_text)


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


def print_timm_model_info(model_name = 'resnet18',input_size = (1, 3, 224, 224)):
    '''
    打印timm模型信息
    :param model_name:
    :param input_size:
    :return:
    '''
    model = timm.create_model(model_name, pretrained=True)
    model_info_print(model, input_size, model_name)
    return model


def temp():
    # 创建模型对象
    # model = Net()
    model_name = 'resnet18'
    model = timm.create_model(model_name, pretrained=True)
    # model_name = 'resnet18'
    # model_name = 'mixnet_s'
    #
    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    # import torchvision.models as models
    # import torchinfo
    #
    # # Load ResNet-18 model from Torch Hub
    # model = models.resnet18(pretrained=False)
    #
    # # Load state dict from URL
    # state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    #
    # # Load state dict into model
    # model.load_state_dict(state_dict)

    # model = timm.create_model(model_name, pretrained=True)
    # 生成随机输入
    # inputs = torch.randn(1, 1, 28, 28)
    # inputs = torch.randn(1, 1, 28, 28)
    # input_size = (3, 224, 224)
    input_size = (1, 3, 224, 224)
    # inputs = (3, 240, 240)

    # inputs = torch.randn(3, 224, 224)
    # 绘制网络结构图
    # dot = make_dot(model(inputs), params=dict(model.named_parameters()))
    # graph = pydotplus.graph_from_dot_data(dot.source)
    # graph.write_png('network.png')

    # if model_name == 'alexnet':
    #     model = models.alexnet(pretrained=True)
    # elif model_name == 'convnext_tiny':
    #     model = models.convnext.convnext_tiny(pretrained=True)
    # elif model_name == 'densenet121':
    #     model = models.densenet.densenet121(pretrained=True)
    # elif model_name == 'efficientnet_b0':
    #     model = models.efficientnet.efficientnet_b0(pretrained=True)
    # elif model_name == 'googlenet':
    #     model = models.googlenet(pretrained=True)
    # elif model_name == 'inception_v3':
    #     model = models.inception.inception_v3(pretrained=True)
    # elif model_name == 'mnasnet0_5':
    #     model = models.mnasnet.mnasnet0_5(pretrained=True)
    # elif model_name == 'mobilenet_v2':
    #     model = models.mobilenet.mobilenet_v2(pretrained=True)
    # elif model_name == 'regnet_y_400mf':
    #     model = models.regnet.regnet_y_400mf(pretrained=True)
    # elif model_name == 'resnet18':
    #     model = models.resnet.resnet18(pretrained=True)
    # elif model_name == 'shufflenet_v2_x0_5':
    #     model = models.shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
    # elif model_name == 'squeezenet1_0':
    #     model = models.squeezenet.squeezenet1_0(pretrained=True)
    # elif model_name == 'vgg16':
    #     model = models.vgg.vgg16(pretrained=True)
    # elif model_name == 'vit_b_16':
    #     model = models.vision_transformer.vit_b_16(pretrained=True)
    # elif model_name == 'swin_t':
    #     model = models.swin_transformer.swin_t(pretrained=True)
    # elif model_name == 'maxvit':
    #     model = models.maxvit.maxvit_t(pretrained=True)
    # elif model_name == 'RetinaNet':
    #     # model = models.detection.RetinaNet(pretrained=True, backbone='resnet50_fpn', num_classes=91)
    #     model = models.detection.retinanet_resnet50_fpn(pretrained=True)

    # model = models.vision_transformer.vit_b_16(pretrained=True)
    # 读取模型摘要信息文件并打印
    # model_summary_print(model,inputs,model_name)
    # model = ConvNet()
    # batch_size = 1
    # input_size=(batch_size, 3, 224, 224)
    # input_size=(batch_size, 3, 224, 224)

    # model_name = 'FasterRCNN'
    # model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model_name = 'r3d_18'
    # model = models.video.r3d_18(pretrained=True)
    # print(model)
    # model_name = 'ssdlite320_mobilenet_v3_large'
    # model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    # summary(model, input_size=(1, 3, 224, 224))

    # summary(model, input_size=(batch_size, 3, 224, 224))
    model_info_print(model, input_size, model_name=model_name)


if __name__ == '__main__':
    print_timm_model_info()