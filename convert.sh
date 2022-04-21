export CUDA_VISIBLE_DEVICES='0'
config=$1
echo $config
CONFIG_FILE=./work_dirs/$config/$config.py
CHECKPOINT_FILE=./work_dirs/$config/latest.pth
OUTPUT_FILE=./work_dirs/$config/latest.onnx
FEATURE_ONNX=./work_dirs/$config/feature.onnx
FEATURE_ENGINE=./work_dirs/$config/feature.engine
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

python feature_cluster/onnx_converter.py $OUTPUT_FILE $FEATURE_ONNX

trtexec \
        --onnx=$OUTPUT_FILE --fp16 \
        --saveEngine=$ENGINE_FILE \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:64x3x224x224


trtexec \
        --onnx=$FEATURE_ONNX --fp16 \
        --saveEngine=$FEATURE_ENGINE \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:64x3x224x224
