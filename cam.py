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
        default='model.backbone.layer4.1.conv3',
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


def load_model(config_path, device):
    cfg = Config.fromfile(config_path)
    dir = os.path.dirname(config_path)
    checkpoint = os.path.join(dir, 'latest.pth')

    # build the model from a config file and a checkpoint file
    model = init_model(cfg, checkpoint, device=device)

    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)

    return model


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape  # 获取feature_conv特征的尺寸
    num_classes, num_channels = weight_softmax.shape
    assert nc == num_channels

    cams = np.einsum('bchw,lc->blhw', feature_conv, weight_softmax)
    return cams



if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.config, args.device).eval()

    feature_layer = get_layer(args.target_layer, model)
    fc_layer = get_layer('model.head.fc', model)

    features = {}


    def hook_feature(module, input, output):  # input是注册层的输入 output是注册层的输出
        features['feature_map'] = output.data.cpu().numpy()


    feature_layer.register_forward_hook(hook_feature)

    weight_softmax = list(fc_layer.parameters())[0].data.numpy()

    for i in range(3):
        dummy = np.random.uniform(0, 1, (1, 3, 224, 224)).astype(np.float32)
        dummy = torch.from_numpy(dummy).to(args.device)
        with torch.no_grad():
            result = model(dummy)
            print(result)
            print(features['feature_map'][0][0][0])
            cams = returnCAM(features['feature_map'],weight_softmax,1)
            print(cams.shape)
