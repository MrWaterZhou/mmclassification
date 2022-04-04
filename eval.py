import argparse
from typing import Tuple, List
import os

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import glob
from threading import Thread
from queue import Queue
import cv2
import time
import json

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def is_fixed(shape: Tuple[int]):
    return not is_dynamic(shape)


def is_dynamic(shape: Tuple[int]):
    return any(dim is None or dim < 0 for dim in shape)


def setup_binding_shapes(
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        host_inputs: List[np.ndarray],
        input_binding_idxs: List[int],
        output_binding_idxs: List[int],
):
    # Explicitly set the dynamic input shapes, so the dynamic output
    # shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, host_input.shape)

    assert context.all_binding_shapes_specified

    host_outputs = []
    device_outputs = []
    for binding_index in output_binding_idxs:
        output_shape = context.get_binding_shape(binding_index)
        # Allocate buffers to hold output results after copying back to host
        buffer = np.empty(output_shape, dtype=np.float32)
        host_outputs.append(buffer)
        # Allocate output buffers on device
        device_outputs.append(cuda.mem_alloc(buffer.nbytes))

    return host_outputs, device_outputs


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    # Calculate start/end binding indices for current context's profile
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile
    print("Engine/Binding Metadata")
    print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
    print("\tNumber of bindings per profile: {}".format(num_bindings_per_profile))
    print("\tFirst binding for profile {}: {}".format(profile_index, start_binding))
    print("\tLast binding for profile {}: {}".format(profile_index, end_binding - 1))

    # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)

    return input_binding_idxs, output_binding_idxs


def load_engine(filename: str):
    # Load serialized engine file into memory
    with open(filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def get_random_inputs(
        engine: trt.ICudaEngine,
        context: trt.IExecutionContext,
        input_binding_idxs: List[int],
        seed: int = 42,
):
    # Input data for inference
    host_inputs = []
    print("Generating Random Inputs")
    print("\tUsing random seed: {}".format(seed))
    np.random.seed(seed)
    for binding_index in input_binding_idxs:
        # If input shape is fixed, we'll just use it
        input_shape = context.get_binding_shape(binding_index)
        input_name = engine.get_binding_name(binding_index)
        print("\tInput [{}] shape: {}".format(input_name, input_shape))
        # If input shape is dynamic, we'll arbitrarily select one of the
        # the min/opt/max shapes from our optimization profile
        if is_dynamic(input_shape):
            print('\ndynamic\n')
            profile_index = context.active_optimization_profile
            profile_shapes = engine.get_profile_shape(profile_index, binding_index)
            print("\tProfile Shapes for [{}]: [kMIN {} | kOPT {} | kMAX {}]".format(input_name, *profile_shapes))
            # 0=min, 1=opt, 2=max, or choose any shape, (min <= shape <= max)
            input_shape = (2, 3, 224, 224)
            print("\tInput [{}] shape was dynamic, setting inference shape to {}".format(input_name, input_shape))

        host_inputs.append(np.random.random(input_shape).astype(np.float32))

    return host_inputs


class ModelLeak:
    contexts = {}

    def __init__(self, enging_path: str, max_batch_size: int, h: int = 224, w: int = 224):
        trt.init_libnvinfer_plugins(None, "")
        self.engine = load_engine(enging_path)
        self.h = h
        self.w = w

    def create_or_get_context(self, batch_size):
        if batch_size not in self.contexts:
            context = self.engine.create_execution_context()
            context.active_optimization_profile = 0
            input_binding_idxs, output_binding_idxs = get_binding_idxs(
                self.engine, context.active_optimization_profile
            )
            input_names = [self.engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]
            output_names = [self.engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]

            max_input = np.random.random((batch_size, 3, self.h, self.w)).astype('float32')

            device_inputs = [cuda.mem_alloc(max_input.nbytes)]
            host_outputs, device_outputs = setup_binding_shapes(
                self.engine, context, [max_input], input_binding_idxs, output_binding_idxs,
            )
            self.contexts[batch_size] = (device_inputs, device_outputs, context, host_outputs)
        return self.contexts[batch_size]

    def infer(self, batched_image: np.ndarray):
        batch_size = batched_image.shape[0]
        device_inputs, device_outputs, context, host_outputs = self.create_or_get_context(batch_size)
        # host_outputs = self.host_outputs
        for h_input, d_input in zip([batched_image], device_inputs):
            cuda.memcpy_htod(d_input, h_input)

        bindings = device_inputs + device_outputs

        context.execute_v2(bindings)

        for h_output, d_output in zip(host_outputs, device_outputs):
            cuda.memcpy_dtoh(h_output, d_output)

        return [x[:batch_size] for x in host_outputs]


class Model:
    contexts = {}

    def __init__(self, enging_path: str, max_batch_size: int, h: int = 224, w: int = 224):
        trt.init_libnvinfer_plugins(None, "")
        self.engine = load_engine(enging_path)
        self.h = h
        self.w = w
        self.device_inputs, self.device_outputs, self.context, self.host_outputs = self.create_or_get_context(
            max_batch_size)

    def create_or_get_context(self, batch_size):
        if batch_size not in self.contexts:
            context = self.engine.create_execution_context()
            context.active_optimization_profile = 0
            input_binding_idxs, output_binding_idxs = get_binding_idxs(
                self.engine, context.active_optimization_profile
            )
            input_names = [self.engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]
            output_names = [self.engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]

            max_input = np.random.random((batch_size, 3, self.h, self.w)).astype('float32')

            device_inputs = [cuda.mem_alloc(max_input.nbytes)]
            host_outputs, device_outputs = setup_binding_shapes(
                self.engine, context, [max_input], input_binding_idxs, output_binding_idxs,
            )
            self.contexts[batch_size] = (device_inputs, device_outputs, context, host_outputs)
        return self.contexts[batch_size]

    def infer(self, batched_image: np.ndarray):
        print(batched_image.shape)
        batch_size = batched_image.shape[0]
        # device_inputs, device_outputs, context, host_outputs = self.create_or_get_context(batch_size)
        # host_outputs = self.host_outputs
        self.context.set_binding_shape(0, batched_image.shape)
        for h_input, d_input in zip([batched_image], self.device_inputs):
            cuda.memcpy_htod(d_input, h_input)

        bindings = self.device_inputs + self.device_outputs

        self.context.execute_v2(bindings)

        for h_output, d_output in zip(self.host_outputs, self.device_outputs):
            cuda.memcpy_dtoh(h_output, d_output)

        return [x[:batch_size] for x in self.host_outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", required=True, type=str,
                        help="Path to TensorRT engine file.")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    trt.init_libnvinfer_plugins(None, "")
    # Load a serialized engine into memory
    engine = load_engine(args.engine)
    print("Loaded engine: {}".format(args.engine))
    # Create context, this can be re-used
    context = engine.create_execution_context()
    # Profile 0 (first profile) is used by default
    context.active_optimization_profile = 0
    print("Active Optimization Profile: {}".format(context.active_optimization_profile))

    # These binding_idxs can change if either the context or the
    # active_optimization_profile are changed
    input_binding_idxs, output_binding_idxs = get_binding_idxs(
        engine, context.active_optimization_profile
    )
    input_names = [engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]

    # Generate random inputs based on profile shapes
    host_inputs = get_random_inputs(engine, context, input_binding_idxs, seed=args.seed)

    # Allocate device memory for inputs. This can be easily re-used if the
    # input shapes don't change
    # TODO: test
    device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]
    # Copy host inputs to device, this needs to be done for each new input
    for h_input, d_input in zip(host_inputs, device_inputs):
        cuda.memcpy_htod(d_input, h_input)

    print("Input Metadata")
    print("\tNumber of Inputs: {}".format(len(input_binding_idxs)))
    print("\tInput Bindings for Profile {}: {}".format(context.active_optimization_profile, input_binding_idxs))
    print("\tInput names: {}".format(input_names))
    print("\tInput shapes: {}".format([inp.shape for inp in host_inputs]))

    # This needs to be called everytime your input shapes change
    # If your inputs are always the same shape (same batch size, etc.),
    # then you will only need to call this once
    host_outputs, device_outputs = setup_binding_shapes(
        engine, context, host_inputs, input_binding_idxs, output_binding_idxs,
    )
    output_names = [engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]

    print("Output Metadata")
    print("\tNumber of Outputs: {}".format(len(output_binding_idxs)))
    print("\tOutput names: {}".format(output_names))
    print("\tOutput shapes: {}".format([out.shape for out in host_outputs]))
    print("\tOutput Bindings for Profile {}: {}".format(context.active_optimization_profile, output_binding_idxs))

    # Bindings are a list of device pointers for inputs and outputs
    bindings = device_inputs + device_outputs

    # Inference
    context.execute_v2(bindings)

    # Copy outputs back to host to view results
    for h_output, d_output in zip(host_outputs, device_outputs):
        cuda.memcpy_dtoh(h_output, d_output)

    # View outputs
    print("Inference Outputs:", host_outputs)

    # Cleanup (Can also use context managers instead)
    del context
    del engine


class Runner:
    def __init__(self, arch, max_batch_size, engine_path, labels, image_queue: Queue, save_path: str):
        assert arch in {'regnetx', 'resnet'}
        self.mean = 0
        self.std = 1
        self.BGR = False
        if arch == 'regnetx':
            self.BGR = True
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.model = Model(engine_path, max_batch_size, 224, 224)
        self.labels = labels
        self.image_queue = image_queue
        self.save_file = open(save_path, 'w')
        self.max_batch_size = max_batch_size
        self.TP = {x: 0 for x in self.labels}
        self.TN = {x: 0 for x in self.labels}
        self.FP = {x: 0 for x in self.labels}
        self.FN = {x: 0 for x in self.labels}

    def preprocess(self, images):
        images = np.concatenate(images, 0)
        if self.BGR:
            images = images[:, :, :, ::-1]
        images = (images - self.mean) / self.std
        images = np.transpose(images, (0, 3, 1, 2))
        images = np.ascontiguousarray(images, dtype=np.float32)

        return images.astype('float32')

    def run(self):
        try:
            filenames = []
            images = []
            filename, image = self.image_queue.get()
            filenames.append(filename)
            images.append(image)
            while (len(filenames) < self.max_batch_size - 1) and self.image_queue.qsize() != 0:
                filename, image = self.image_queue.get()
                filenames.append(filename)
                images.append(image)

            images = self.preprocess(images)
            results = self.model.infer(images)[0]
            r_list = []
            for filename, res in zip(filenames, results):
                tags = {'image': filename['image'], 'TP': [], 'TN': [], 'FP': [], 'FN': []}
                for r, label in zip(res, self.labels):
                    if r > 0.5:
                        if filename[label] == 1:
                            self.TP[label] += 1
                            tags['TP'].append(label)
                        else:
                            self.FP[label] += 1
                            tags['FP'].append(label)
                    else:
                        if filename[label] == 1:
                            self.FN[label] += 1
                            tags['FN'].append(label)
                        else:
                            self.TN[label] += 1
                            tags['TN'].append(label)
                if len(tags['TP'] + tags['TN'] + tags['FP'] + tags['FN']) > 0:
                    r_list.append(json.dumps(tags, ensure_ascii=False))
            for result in r_list:
                self.save_file.write(result + '\n')
        except Exception as e:
            print(e.__str__())
        return 'done'


class Loader(Thread):
    def __init__(self, file_list: List, image_queue: Queue):
        super().__init__()
        self.file_list = file_list
        self.image_queue = image_queue
        self.end = False

    def run(self) -> None:
        for file in self.file_list:
            file = json.loads(file.strip)
            try:
                image = cv2.imread(file['image'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.expand_dims(image, 0)
                self.image_queue.put((file, image))
            except:
                pass
        self.end = True


def parse_args():
    parser = argparse.ArgumentParser(description='batch predict with tensorrt')
    parser.add_argument('engine_path', help='engine path')
    parser.add_argument('source')
    parser.add_argument('save_path')
    parser.add_argument('--num_preprocess_threads', default=8)
    parser.add_argument('--arch', default='regnetx')
    parser.add_argument('--max_batch_size', default=64, type=int)
    parser.add_argument('--labels', default='0,1')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # config
    num_preprocess_threads = args.num_preprocess_threads
    labels = ['性感_胸部', '色情_女胸', '色情_男下体', '色情_口交', '性感_内衣裤', '性感_男性胸部', '色情_裸露下体', '性感_腿部特写', '正常']
    image_queue = Queue(100)
    file_queue = Queue(100)
    runner = Runner(args.arch, args.max_batch_size, args.engine_path, labels, image_queue,
                    args.save_path)

    if os.path.isfile(args.source):
        files = open(args.source).readlines()
    else:
        files = glob.glob(args.source)

    chunk_size = len(files) // num_preprocess_threads
    ts = []
    for i in range(num_preprocess_threads):
        loader = Loader(files[i * chunk_size: (i + 1) * chunk_size], image_queue)
        loader.daemon = True
        loader.start()
        ts.append(loader)
    import os

    status = sum([int(t.end) for t in ts]) < len(ts)
    while (file_queue.qsize() > 0) or status:
        runner.run()
        status = sum([int(t.end) for t in ts]) < len(ts)

    for l in labels:
        precision = runner.TP[l] / (runner.TP[l] + runner.FP[l])
        recall = runner.TP[l] / (runner.TP[l] + runner.FN[l])
        print('{}, precision:{}, recall:{}'.format(l, precision, recall))

    print('done')
