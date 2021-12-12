import argparse
import os
from mmcv import Config, DictAction
from mmcls.apis import init_model
from functools import partial


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


if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.config, args.device)

    feature_layer = get_layer(args.target_layer, model)
    fc_layer = get_layer('model.backbone.head.fc', model)

    print(model)
    print(fc_layer)
    print(fc_layer.parameters())
