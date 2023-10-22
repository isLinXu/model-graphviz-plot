import json
import timm
import torchvision.models as models

def get_timm_model_list():
    timm_model_list = timm.list_models()
    timm_model_list = sorted(timm_model_list)
    print("timm_model_list: ", timm_model_list)
    print("len(timm_model_list): ", len(timm_model_list))
    with open("timm_model_list.json", "w") as json_file:
        json.dump(timm_model_list, json_file)

def get_torchvision_model_list():
    torchvision_model_list = dir(models)
    torchvision_model_list = sorted(torchvision_model_list)
    print("torchvision_model_list: ", torchvision_model_list)
    print("len(torchvision_model_list): ", len(torchvision_model_list))
    with open("torchvision_model_list.json", "w") as json_file:
        json.dump(torchvision_model_list, json_file)


if __name__ == '__main__':
    get_timm_model_list()
    get_torchvision_model_list()