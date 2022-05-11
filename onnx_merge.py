import onnx
import numpy as np
from onnx import numpy_helper
from onnx import helper


def get_initializer_value(graph, name):
    status = 0
    for node in graph.initializer:
        if node.name == name:
            status = 1
            break
    if status == 0:
        return None
    else:
        return numpy_helper.to_array(node)


def get_initializer_node(graph, name):
    status = 0
    for node in graph.initializer:
        if node.name == name:
            status = 1
            break
    if status == 0:
        return None
    else:
        return node


def merge_fc_head(model_1, model_2, index=None, path='model.onnx'):
    graph_1 = model_1.graph
    graph_2 = model_2.graph
    weight_1 = get_initializer_value(graph_1, 'head.fc.weight')
    weight_2 = get_initializer_value(graph_2, 'head.fc.weight')
    weight = np.concatenate([weight_1, weight_2])
    if index is None:
        index = list(range(weight.shape[0]))
    weight = weight[index, :]

    bias_1 = get_initializer_value(graph_1, 'head.fc.bias')
    bias_2 = get_initializer_value(graph_2, 'head.fc.bias')
    bias = np.concatenate([bias_1, bias_2])
    bias = bias[index]

    weight_node_new = numpy_helper.from_array(weight, 'head.fc.weight')
    bias_node_new = numpy_helper.from_array(bias, 'head.fc.bias')

    weight_node = get_initializer_node(graph_1, 'head.fc.weight')
    bias_node = get_initializer_node(graph_1, 'head.fc.bias')

    # update weight
    while len(weight_node.dims):
        weight_node.dims.pop()
    for d in weight_node_new.dims:
        weight_node.dims.append(d)

    weight_node.raw_data = weight_node_new.raw_data

    # update bias
    while len(bias_node.dims):
        bias_node.dims.pop()
    for d in bias_node_new.dims:
        bias_node.dims.append(d)
    bias_node.raw_data = bias_node_new.raw_data

    # change output shape
    graph_1.output[0].type.tensor_type.shape.dim[1].dim_value = weight.shape[0]
    # onnx.checker.check_model(model_1)
    onnx.save(model_1, path)


if __name__ == '__main__':
    import sys

    model_1 = onnx.load_model(sys.argv[1])
    model_2 = onnx.load_model(sys.argv[2])
    index = sys.argv[3].split(',')
    index = [int(x) for x in index]
    save_path = sys.argv[4]

    merge_fc_head(model_1, model_2, index, save_path)
