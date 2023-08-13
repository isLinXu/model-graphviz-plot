
SAMPLE_IMAGE=000000439715.jpg
CONFIG_FILE=configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml
MODEL_WEIGHTS=weights/COCO-Detection/faster_rcnn_R_101_C4_3x/model_final_298dad.pkl
EXPORT_METHOD=tracing
FORMAT=onnx
OUTPUT_DIR=./
DEVICE=cpu

python tools/deploy/export_model.py \
    --sample-image $SAMPLE_IMAGE  \
    --config-file $CONFIG_FILE \
    --export-method $EXPORT_METHOD \
    --format $FORMAT \
    --output $OUTPUT_DIR \
    MODEL.WEIGHTS $MODEL_WEIGHTS \
    MODEL.DEVICE $DEVICE