import onnx
from onnx import helper
import sys

model = onnx.load(sys.argv[1])
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = '696'
model.graph.output.pop()
model.graph.output.append(intermediate_layer_value_info)
print(model.graph.output[0])
onnx.save(model, sys.argv[2])
