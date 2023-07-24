import timm
import torchvision.models as models
from torchviz import make_dot, make_dot_from_trace
import torch

import os
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin/dot'

def visualize_timm_model(model_name, input_size=(1, 3, 224, 224), visual_tag=False):
    '''
    visual timm model
    :param model_name:
    :param input_size:
    :param visual_tag:
    :return:
    '''
    model = timm.create_model(model_name, pretrained=True)
    print(model)
    model.eval()
    x = torch.randn(input_size)
    with torch.no_grad():
        y = model(x)
    vis_graph = make_dot(y.mean(), params=dict(model.named_parameters()))
    vis_graph.render(model_name, format='png')
    if visual_tag:
        vis_graph.view()

def visual_torchvision_model(model_name, input_size=(1, 3, 224, 224), visual_tag=False):
    '''
    visual torchvision model
    :param model_name:
    :param input_size:
    :param visual_tag:
    :return:
    '''
    global model
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name == 'convnext_tiny':
        model = models.convnext.convnext_tiny(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet.densenet121(pretrained=True)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet.efficientnet_b0(pretrained=True)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception.inception_v3(pretrained=True)
    elif model_name == 'mnasnet0_5':
        model = models.mnasnet.mnasnet0_5(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet.mobilenet_v2(pretrained=True)
    elif model_name == 'regnet_y_400mf':
        model = models.regnet.regnet_y_400mf(pretrained=True)
    elif model_name == 'resnet18':
        model = models.resnet.resnet18(pretrained=True)
    elif model_name == 'shufflenet_v2_x0_5':
        model = models.shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
    elif model_name == 'squeezenet1_0':
        model = models.squeezenet.squeezenet1_0(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg.vgg16(pretrained=True)
    elif model_name == 'vit_b_16':
        model = models.vision_transformer.vit_b_16(pretrained=True)
    elif model_name == 'swin_t':
        model = models.swin_transformer.swin_t(pretrained=True)
    elif model_name == 'maxvit':
        model = models.maxvit.maxvit_t(pretrained=True)
    else:
        raise NotImplementedError
    print(model)
    model.eval()
    x = torch.randn(input_size)

    y = model(x)
    vis_graph = make_dot(y, params=dict(model.named_parameters()))
    # vis_graph = make_dot(y.mean(), params=dict(model.named_parameters()))
    vis_graph.render(model_name, format='png')
    if visual_tag:
        vis_graph.view()


if __name__ == '__main__':
    model_name = 'mobilenet_v2'
    # input_size = (4, 3, 32, 32)
    input_size = (1, 3, 224, 224)
    visual_torchvision_model(model_name, input_size, visual_tag=False)
