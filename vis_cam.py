# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import math
import sys
from pathlib import Path
import cv2
import os
import torch
import mmcv
import numpy as np
from mmcv import Config, DictAction
from threading import Thread
from queue import Queue
from typing import Tuple, List

from mmcls.apis import init_model
from mmcls.datasets.pipelines import Compose
from mmcls.models.backbones import SwinTransformer, T2T_ViT, VisionTransformer

try:
    from pytorch_grad_cam import (EigenCAM, GradCAM, GradCAMPlusPlus, XGradCAM,
                                  EigenGradCAM, LayerCAM)
    import pytorch_grad_cam.activations_and_gradients as act_and_grad
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError(
        'please use `pip install grad-cam` to install pytorch_grad_cam')

# set of transforms, which just change data format, not change the pictures
FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize', 'ImageToTensor', 'Collect'}

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}

# Transformer set based on ViT
ViT_based_Transformers = tuple([T2T_ViT, VisionTransformer])

# Transformer set based on Swin
Swin_based_Transformers = tuple([SwinTransformer])


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--source', help='Image file')
    parser.add_argument(
        '--target-layers',
        default=['model.backbone.layer4.0'],
        nargs='+',
        type=str,
        help='The target layers to get CAM')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
             f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=None,
        type=int,
        help='The target category to get CAM, default to use result '
             'get from given model')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
             '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The path to save visualize cam image, default not to save.')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


def build_reshape_transform(model):
    """build reshape_transform for `cam.activations_and_grads`, some neural
    networks such as SwinTransformer and VisionTransformer need an additional
    reshape operation.

    CNNs don't need, jush return None
    """
    if isinstance(model.backbone, Swin_based_Transformers):
        has_clstoken = False
    elif isinstance(model.backbone, ViT_based_Transformers):
        has_clstoken = True
    else:
        return None

    def _reshape_transform(tensor, has_clstoken=has_clstoken):
        """reshape_transform helper."""
        tensor = tensor[:, 1:, :] if has_clstoken else tensor
        # get heat_map_height and heat_map_width, preset input is a square
        heat_map_size = tensor.size()[1]
        height = width = int(math.sqrt(heat_map_size))
        message = 'Only support input pictures with the same length and ' \
                  'width when using Tansformer neural networks.'
        assert height * height == heat_map_size, message
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return _reshape_transform


def apply_transforms(img_path, pipeline_cfg):
    """Since there are some transforms, which will change the regin to
    inference such as CenterCrop.So it is necessary to get inference image."""
    data = dict(img_info=dict(filename=img_path), img_prefix=None)

    def split_pipeline_cfg(pipeline_cfg):
        """to split the transfoms into image_transforms and
        format_transforms."""
        image_transforms_cfg, format_transforms_cfg = [], []
        if pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        for transform in pipeline_cfg:
            if transform['type'] in FORMAT_TRANSFORMS_SET:
                format_transforms_cfg.append(transform)
            else:
                image_transforms_cfg.append(transform)
        return image_transforms_cfg, format_transforms_cfg

    image_transforms, format_transforms = split_pipeline_cfg(pipeline_cfg)
    image_transforms = Compose(image_transforms)
    format_transforms = Compose(format_transforms)

    intermediate_data = image_transforms(data)
    inference_img = copy.deepcopy(intermediate_data['img'])
    format_data = format_transforms(intermediate_data)

    return format_data, inference_img


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object and the get_loss
    function."""

    class mmActivationsAndGradients(act_and_grad.ActivationsAndGradients):
        """since the original __call__ can not pass additional parameters we
        modify the function to return torch.tensor."""

        def __call__(self, x):
            self.gradients = []
            self.activations = []

            return self.model(x, return_loss=False, post_processing=False)

    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    cam.activations_and_grads = mmActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model lyaer from given str."""
    cur_layer = model
    assert layer_str.startswith(
        'model'), "target-layer must start with 'model'"
    layer_items = layer_str.strip().split('.')
    assert not (layer_items[-1].startswith('relu')
                or layer_items[-1].startswith('bn')
                ), "target-layer can't be 'bn' or 'relu'"
    for item_str in layer_items[1:]:
        if hasattr(cur_layer, item_str):
            cur_layer = getattr(cur_layer, item_str)
        else:
            raise ValueError(
                f"model don't have `{layer_str}`, please use valid layers")
    return cur_layer


def show_cam_grad(grayscale_cam, src_img, title, out_path=None, origin_image=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    if origin_image is None:
        src_img = np.float32(src_img) / 255
    else:
        src_img = np.float32(origin_image) / 255
        shape = origin_image.shape
        grayscale_cam = cv2.resize(grayscale_cam, (shape[1], shape[0]))
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        mmcv.imwrite(visualization_img, str(out_path))
    else:
        mmcv.imshow(visualization_img, win_name=title)


class Loader(Thread):
    def __init__(self, file_list: List, image_queue: Queue):
        super().__init__()
        self.file_list = file_list
        self.image_queue = image_queue
        self.end = False

    def run(self) -> None:
        for file in self.file_list:
            try:
                image = cv2.imread(file.strip())
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.expand_dims(image, 0)
                self.image_queue.put((file, image))
            except:
                pass
        self.end = True


class Runner:
    def __init__(self, args, cam_model, max_batch_size, image_queue: Queue, save_queue: Queue, arch='regnetx',
                 device='cpu'):
        assert arch in {'regnetx', 'resnet'}
        self.mean = 0
        self.std = 1
        self.BGR = False
        if arch == 'regnetx':
            self.BGR = True
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.model = cam_model
        self.image_queue = image_queue
        self.max_batch_size = max_batch_size
        self.device = device
        self.args = args
        self.save_queue = save_queue

    def preprocess(self, images):
        images = np.concatenate(images, 0)
        if self.BGR:
            images = images[:, :, :, ::-1]
        images = (images - self.mean) / self.std
        images = np.transpose(images, (0, 3, 1, 2))
        images = np.ascontiguousarray(images, dtype=np.float32)

        return torch.tensor(images, device=self.device)

    def run(self):
        try:
            filenames = []
            shapes = []
            images = []
            filename, image, shape = self.image_queue.get()
            filenames.append(filename)
            images.append(image)
            shapes.append(shape)
            while (len(filenames) < self.max_batch_size - 1) and self.image_queue.qsize() != 0:
                filename, image, shape = self.image_queue.get()
                filenames.append(filename)
                images.append(image)
                shapes.append(shape)

            images = self.preprocess(images)

            grayscale_cams = self.model(
                input_tensor=images,
                target_category=self.args.target_category,
                eigen_smooth=self.args.eigen_smooth,
                aug_smooth=self.args.aug_smooth)

            for filename, shape, grayscale_cam in zip(filenames, shapes, grayscale_cams):
                self.save_queue.put((filename, shape, grayscale_cam))

        except Exception as e:
            print(e.__str__())
        return 'done'


class Saver(Thread):
    def __init__(self, save_path: str, save_queue: Queue):
        super().__init__()
        self.save_path = save_path
        self.save_queue = save_queue

    def run(self) -> None:
        while True:
            filename, shape, grayscale_cam = self.save_queue.get()
            try:
                grayscale_cam = cv2.resize(grayscale_cam, (shape[1], shape[0]))
                filename = filename.split('/')[-1]
                filename = os.path.join(self.save_path,filename)
                cv2.imwrite(filename, grayscale_cam * 255)
            except Exception as e:
                print(e)



def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        sys.exit()

    # build target layers
    target_layers = [
        get_layer(layer_str, model) for layer_str in args.target_layers
    ]
    assert len(args.target_layers) != 0, '`--target-layers` can not be empty'

    # init a cam grad calculator
    use_cuda = True if 'cuda' in args.device else False
    reshape_transform = build_reshape_transform(model)
    cam = init_cam(args.method, model, target_layers, use_cuda,
                   reshape_transform)

    if args.img is not None:
        # apply transform and perpare data
        image = cv2.imread(args.img)
        shape = image.shape
        data, src_img = apply_transforms(args.img, cfg.data.test.pipeline)
        data['img'] = data['img'].unsqueeze(0)

        # calculate cam grads and show|save the visualization image
        grayscale_cam = cam(
            input_tensor=data['img'],
            target_category=args.target_category,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth)
        show_cam_grad(
            grayscale_cam, src_img, title=args.method, out_path=args.save_path, origin_image=image)


    elif args.source is not None:
        image_queue = Queue(100)
        save_queue = Queue(100)

        num_preprocess_threads = 8
        max_batch_size=32
        num_postprocess_threrads = 4

        runner = Runner(args, cam, max_batch_size, image_queue, save_queue, arch='regnetx',
                 device=args.device)


        import glob
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

        for i in range(num_postprocess_threrads):
            saver = Saver(args.save_path,save_queue)
            saver.daemon = True
            saver.start()

        status = sum([int(t.end) for t in ts]) < len(ts)
        while (image_queue.qsize() > 0) or status:
            runner.run()
            status = sum([int(t.end) for t in ts]) < len(ts)

        import time
        while save_queue.qsize() > 0:
            time.sleep(10)

        sys.exit()








if __name__ == '__main__':
    main()
