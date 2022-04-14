import onnx
from onnx import helper
import sys

model = onnx.load(sys.argv[1])
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = '696'
intermediate_layer_value_info.type.tensor_type.elem_type = 1
intermediate_layer_value_info.type.tensor_type.shape.dim.append(model.graph.output[0].type.tensor_type.shape.dim[0])
intermediate_layer_value_info.type.tensor_type.shape.dim.append(model.graph.output[0].type.tensor_type.shape.dim[1])
intermediate_layer_value_info.type.tensor_type.shape.dim[1].dim_value = 1360

model.graph.output.append(intermediate_layer_value_info)
print(model.graph.output[0], model.graph.output[1])
onnx.save(model, sys.argv[2])
