# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from functools import partial

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
import cv2
from mmcls.models import build_classifier
from onnx import shape_inference, helper, TensorProto
import onnx
torch.manual_seed(3)


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(
        low=0, high=num_classes, size=(N, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2onnx(model,
                 input_shape,
                 opset_version=11,
                 dynamic_export=False,
                 show=False,
                 output_file='tmp.onnx',
                 do_simplify=False,
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if hasattr(model.head, 'num_classes'):
        num_classes = model.head.num_classes
    # Some backbones use `num_classes=-1` to disable top classifier.
    elif getattr(model.backbone, 'num_classes', -1) > 0:
        num_classes = model.backbone.num_classes
    else:
        raise AttributeError('Cannot find "num_classes" in both head and '
                             'backbone, please check the config file.')

    mm_inputs = _demo_mm_inputs(input_shape, num_classes)

    imgs = mm_inputs.pop('imgs')
    img_list = [img[None, :] for img in imgs]

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(model.forward, img_metas={}, return_loss=False)
    register_extra_symbolics(opset_version)

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                # 2: 'width',
                # 3: 'height'
            },
            'probs': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}
    print(model)
    with torch.no_grad():
        torch.onnx.export(
            model, (img_list,),
            output_file,
            input_names=['input'],
            output_names=['probs'],
            export_params=True,
            keep_initializers_as_inputs=True,
            # TODO
            # training=False,
            do_constant_folding=True,
            # TODO
            dynamic_axes=dynamic_axes,
            verbose=False,
            opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward
    onnx_model = onnx.load_model(output_file)
    print(output_file)
    onnx_module = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_module, output_file+'.shape.onnx')

    if do_simplify:
        from mmcv import digit_version
        import onnxsim

        min_required_version = '0.3.0'
        assert digit_version(mmcv.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        if dynamic_axes:
            input_shape = (input_shape[0], input_shape[1], input_shape[2],
                           input_shape[3])
        else:
            input_shape = (input_shape[0], input_shape[1], input_shape[2],
                           input_shape[3])
        imgs = _demo_mm_inputs(input_shape, model.head.num_classes).pop('imgs')
        input_dic = {'input': imgs.detach().cpu().numpy()}
        input_shape_dic = {'input': list(input_shape)}

        model_opt, check_ok = onnxsim.simplify(
            output_file,
            input_shapes=input_shape_dic,
            input_data=input_dic,
            dynamic_input_shape=dynamic_export)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            print('Failed to simplify ONNX model.')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # test the dynamic model
        z = []
        image = cv2.imread(
            '/home/zhou/dataset/subimages/3b55e030-f440-11eb-91bf-ac1f6b9545ae_1627984227.55004050_0.jpg')
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, 0)
        image = (image - mean) / std
        image = np.transpose(image, (0, 3, 1, 2))
        if dynamic_axes:
            for batch in range(1, 5):
                images = np.concatenate([image] * batch, 0)
                imgs = torch.tensor(images.astype(np.float32))
                img_list = [imgs]

                # check the numerical value
                # get pytorch output
                print(img_list[0].shape)
                pytorch_result = model(img_list, img_metas={}, return_loss=False)

                # get onnx output
                input_all = [node.name for node in onnx_model.graph.input]
                input_initializer = [
                    node.name for node in onnx_model.graph.initializer
                ]
                net_feed_input = list(set(input_all) - set(input_initializer))
                assert (len(net_feed_input) == 1)
                sess = rt.InferenceSession(output_file)
                onnx_result = sess.run(
                    None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]

                z.append(onnx_result[0])
            for zz in z:
                print(zz)
        else:
            for batch in range(1, 2):
                images = np.concatenate([image] * batch, 0)
                imgs = torch.tensor(images.astype(np.float32))
                img_list = [imgs]

                # check the numerical value
                # get pytorch output
                print(img_list[0].shape)
                pytorch_result = model(img_list, img_metas={}, return_loss=False)

                # get onnx output
                input_all = [node.name for node in onnx_model.graph.input]
                input_initializer = [
                    node.name for node in onnx_model.graph.initializer
                ]
                net_feed_input = list(set(input_all) - set(input_initializer))
                assert (len(net_feed_input) == 1)
                sess = rt.InferenceSession(output_file)
                onnx_result = sess.run(
                    None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]

                z.append(onnx_result[0])
            for zz in z:
                print(zz)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMCls to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
                          1,
                          3,
                      ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    classifier = build_classifier(cfg.model)

    if args.checkpoint:
        load_checkpoint(classifier, args.checkpoint, map_location='cpu')

    # convert model to onnx file
    pytorch2onnx(
        classifier,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        dynamic_export=args.dynamic_export,
        output_file=args.output_file,
        do_simplify=args.simplify,
        verify=args.verify)
