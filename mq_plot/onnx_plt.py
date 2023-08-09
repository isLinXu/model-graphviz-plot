import timm
import torch
import netron
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

# model = Net()

def onnx_vis_plot(model_name, model, dummy_input):
    model.eval()
    # 保存模型为 ONNX 格式

    onnx_name = model_name + ".onnx"

    torch.onnx.export(model, dummy_input, onnx_name, verbose=True, input_names=['input'], output_names=['output'])

    # 在浏览器中打开 Netron
    netron.start(onnx_name)


if __name__ == '__main__':

    # model_name = 'dm_nfnet_f0'
    # model = timm.create_model(model_name, pretrained=True)
    model_name = 'vit_b_16'
    # model = models.keypointrcnn_resnet50_fpn(pretrained=True)
    # model = models.quantization.shufflenet_v2_x0_5(pretrained=True)
    model = models.vit_b_16(pretrained=True)
    # model = models.segmentation.squeezenet1_0(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    # dummy_input = torch.randn(1, 10)
    onnx_vis_plot(model_name, model,dummy_input)
