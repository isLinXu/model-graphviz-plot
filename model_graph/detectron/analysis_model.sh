
# Counting the number of parameters in a model
python tools/analyze_model.py --task parameter --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml

# Counting the number of flops in a model
python tools/analyze_model.py --task flop --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml

# Counting the number of activation in a model
python tools/analyze_model.py --task activation --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml

# Counting the number of structure in a model
python tools/analyze_model.py --task structure --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml