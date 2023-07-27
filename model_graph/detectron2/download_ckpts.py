

import yaml

yaml_path = '/Users/gatilin/PycharmProjects/model-graphviz-plot/model_graph/detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml'
# 读取 YAML 文件
with open(yaml_path, 'r') as f:
    yaml_file = yaml.safe_load(f)

# 获取 WEIGHTS 路径
weights_path = yaml_file['MODEL']['WEIGHTS']

# 拼接链接
download_url = 'https://dl.fbaipublicfiles.com/detectron2/' + weights_path.split('detectron2://')[1]

# 输出链接
print(download_url)

import requests

import urllib.request
def download_file(url, save_path,file_name):
    urllib.request.urlretrieve(url, file_name)
    print("文件下载完成！")

# 定义 WEIGHTS_URL 和保存路径
WEIGHTS_URL = download_url
save_path = ""
file_name = "model_final_f10217.pkl"
# 下载文件
download_file(download_url, save_path, file_name)

