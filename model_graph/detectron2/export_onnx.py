
import torch
import torchvision
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch.onnx import export
import netron


# set up configuration
cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.MODEL.DEVICE = "cpu"

# create predictor
predictor = DefaultPredictor(cfg)

# create dummy input
inputs = torch.rand((1, 3, 800, 800))

# export to onnx format
export(
    predictor.model,
    inputs,
    "faster_rcnn_R_50_FPN_3x.onnx",
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["boxes", "scores", "labels"]
)

netron.start("faster_rcnn_R_50_FPN_3x.onnx")