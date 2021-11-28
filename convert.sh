export CUDA_VISIBLE_DEVICES='0'
config=$1
echo $config
CONFIG_FILE=./work_dirs/$config/$config.py
CHECKPOINT_FILE=./work_dirs/$config/latest.pth
OUTPUT_FILE=./work_dirs/$config/latest.onnx
ENGINE_FILE=./work_dirs/$config/model.engine
IMAGE_SHAPE='224 224'
OPSET_VERSION=11
python pytorch2onnx.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --opset-version ${OPSET_VERSION} \
    --show \
    --verify \
    --dynamic-export
trtexec \
        --onnx=$OUTPUT_FILE --fp16 \
        --saveEngine=$ENGINE_FILE \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:32x3x224x224