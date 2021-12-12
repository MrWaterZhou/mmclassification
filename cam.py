import argparse
import os

import torch
from mmcv import Config, DictAction
from mmcls.apis import init_model
from functools import partial
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--source', help='Image file')
    parser.add_argument('--save_path', help='save path')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--target-layer',
        default='model.backbone.layer4.1.relu',
        type=str,
        help='The target layers to get CAM')

    args = parser.parse_args()
    return args


def get_layer(layer_str, model):
    """get model lyaer from given str."""
    cur_layer = model
    assert layer_str.startswith(
        'model'), "target-layer must start with 'model'"
    layer_items = layer_str.strip().split('.')
    for item_str in layer_items[1:]:
        if hasattr(cur_layer, item_str):
            cur_layer = getattr(cur_layer, item_str)
        else:
            raise ValueError(
                f"model don't have `{layer_str}`, please use valid layers")
    return cur_layer


def load_model(config_path, device):
    cfg = Config.fromfile(config_path)
    dir = os.path.dirname(config_path)
    checkpoint = os.path.join(dir, 'latest.pth')

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, checkpoint, device=device)

    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)

    return model


class Cam:
    def __init__(self, model, target_layer, fc_layer_name='model.head.fc', device='cpu'):
        self.model = model
        self.feature_layer = get_layer(target_layer, model)
        fc_layer = get_layer(fc_layer_name, model)
        self.weight_softmax = list(fc_layer.parameters())[0].data.numpy()
        self.features = {}

        def hook_feature(module, input, output):  # input是注册层的输入 output是注册层的输出
            self.features['feature_map'] = output.data.cpu().numpy()

        self.feature_layer.register_forward_hook(hook_feature)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
        self.device = device

    def preprocess(self, images):
        images = np.concatenate(images, 0)
        images = images[:, :, :, ::-1]
        images = (images - self.mean) / self.std
        images = np.transpose(images, (0, 3, 1, 2))
        images = np.ascontiguousarray(images, dtype=np.float32)
        return images

    def get_cam_matrix(self, images, labels=None):
        images = self.preprocess(images)
        images = torch.from_numpy(images).to(self.device)
        preds = self.model(images)  # batch_size, num_classes
        feature_conv = self.features['feature_map']

        # 验证shape
        bz, nc, h, w = feature_conv.shape  # 获取feature_conv特征的尺寸
        num_classes, num_channels = self.weight_softmax.shape
        assert nc == num_channels

        # 批处理归一化
        cams = np.einsum('bchw,lc->blhw', feature_conv, self.weight_softmax)
        # normalize
        min_cams = np.min(cams, axis=(-1, -2), keepdims=True)
        cams = cams - min_cams
        max_cams = np.max(cams, axis=(-1, -2), keepdims=True)
        cams = cams / max_cams  # batch_size, num_classes, h,w
        cams = np.uint8(cams * 255)

        if labels is None:
            xs, ys = np.where(preds > 0.5)
            xs = xs.tolist()
            ys = ys.tolist()
        else:
            xs = []
            ys = []
            for x, label in enumerate(labels):
                xs.extend([x] * len(label))
                ys.extend(label)

        results = [{}] * bz
        for x, y in zip(xs, ys):
            cam = cams[x, y, :, :]
            results[x][y] = cam
        return results


if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.config, args.device).eval()

    cam_model = Cam(model, args.target_layer, device=args.device)

    images = [np.random.uniform(0, 255, (1, 224, 224, 3))]
    cams = cam_model.get_cam_matrix(images, [[0]])
    print(cams)
