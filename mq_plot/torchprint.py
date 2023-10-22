import torch
from torchsummary import summary
import sys
from torchinfo import summary


def model_info_print(model, input_size, model_name):
    '''
    打印模型摘要信息
    :param model:
    :param input_size:
    :param model_name:
    :return:
    '''
    # 重定向sys.stdout到文件
    sys.stdout = open(model_name + '_model_summary.txt', 'w')
    # 打印模型摘要信息
    try:
        # summary(model, input_size)
        summary(model, input_data=input_size)
    except Exception as e:
        try:
            summary(model, input_size=input_size)
        except Exception as e:
            print("Error: ", e)
    # 关闭文件和恢复sys.stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
        # 读取模型摘要信息文件并打印
        print("model_name:", model_name)
    # 读取模型摘要信息文件并打印
    with open(model_name + '_model_summary.txt', 'r') as f:
        summary_text = f.read()
        print(summary_text)



def print_timm_model_info(model_name='resnet18', input_size=(1, 3, 224, 224)):
    '''
    打印timm模型信息
    :param model_name:
    :param input_size:
    :return:
    '''
    import timm
    model = timm.create_model(model_name, pretrained=True)
    model_info_print(model, input_size, model_name)
    return model


def print_torchvision_model_info(model_name='resnet18', input_size=(1, 3, 224, 224)):
    global model
    import torchvision.models as models
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
    elif model_name == 'RetinaNet':
        # model = models.detection.RetinaNet(pretrained=True, backbone='resnet50_fpn', num_classes=91)
        model = models.detection.retinanet_resnet50_fpn(pretrained=True)
    elif model_name == 'FasterRCNN':
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'ssdlite320_mobilenet_v3_large':
        model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
    # summary(model, input_size=(1, 3, 224, 224))
    model_info_print(model, input_size, model_name)


def print_hf_model_info(model_name_or_path, input_data):

    from transformers import AutoTokenizer, AutoModel
    from torchinfo import summary
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    # Print the model summary using torchinfo
    summary(model, input_data=input_data)
    model_info_print(model, input_data, model_name_or_path)


if __name__ == '__main__':
    # 打印timm模型摘要信息
    # print_timm_model_info()

    # 打印torchvision模型摘要信息
    # print_torchvision_model_info()

    # 打印transformers模型摘要信息
    # Create a sample input
    input_data = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    # Print the summary for the GPT-2 model
    print_hf_model_info("gpt2", input_data)

    # Print the summary for the BERT model
    # print_hf_model_info("bert-base-uncased", input_data)