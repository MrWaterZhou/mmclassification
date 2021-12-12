import argparse
import os

import torch
from mmcv import Config, DictAction
from mmcls.apis import init_model
from functools import partial
import numpy as np
import cv2
from threading import Thread
from queue import Queue
from typing import Tuple, List
import json
from pytorch_grad_cam.utils.image import show_cam_on_image


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--source', help='Image file')
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
        with torch.no_grad():
            images = torch.from_numpy(images).to(self.device)
            preds = self.model.forward(images)  # batch_size, num_classes
            preds = np.vstack(preds)
        print(preds.shape)
        feature_conv = self.features['feature_map']
        print(feature_conv.shape)

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
        # cams = np.uint8(cams * 255)

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


class Loader(Thread):
    def __init__(self, file_list: List, image_queue: Queue):
        super().__init__()
        self.file_list = file_list
        self.image_queue = image_queue
        self.end = False
        self.labels = ['性感_胸部', '色情_女胸', '色情_男下体', '色情_口交',
                       '性感_内衣裤', '性感_男性胸部', '色情_裸露下体', '性感_腿部特写']

    def run(self) -> None:
        for file in self.file_list:
            image_path = file['image']
            label_choices = [i for i, label in enumerate(self.labels) if file[label] == 1]
            if len(label_choices) > 0:
                try:
                    image_raw = cv2.imread(image_path)
                    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    image = np.expand_dims(image, 0)
                    self.image_queue.put((file, image, image_raw, label_choices))
                except Exception as e:
                    print(e)
        self.end = True


class Runner:
    def __init__(self, cam_model: Cam, max_batch_size, image_queue: Queue, save_queue: Queue):
        self.cam_model = cam_model
        self.image_queue = image_queue
        self.max_batch_size = max_batch_size
        self.args = args
        self.save_queue = save_queue

    def run(self):
        try:
            filenames = []
            images_raw = []
            images = []
            labels = []
            filename, image, image_raw, label = self.image_queue.get()
            filenames.append(filename)
            images.append(image)
            images_raw.append(image_raw)
            labels.append(label)
            while (len(filenames) < self.max_batch_size - 1) and self.image_queue.qsize() != 0:
                filename, image, image_raw, label = self.image_queue.get()
                filenames.append(filename)
                images.append(image)
                images_raw.append(image_raw)
                labels.append(label)

            grayscale_cams = self.cam_model.get_cam_matrix(images, None)
            print(grayscale_cams)

            for filename, image_raw, grayscale_cam in zip(filenames, images_raw, grayscale_cams):
                self.save_queue.put((filename, image_raw, grayscale_cam))

        except Exception as e:
            print(e.__str__())
        return 'done'


class Saver(Thread):
    def __init__(self, save_queue: Queue, result_queue: Queue):
        super().__init__()
        self.save_queue = save_queue
        self.result_queue = result_queue

    def paste_color_block(self, image, grayscale_cam, shape):
        image = image.copy()

        cam_bin = (grayscale_cam > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours((cam_bin * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            center, r = cv2.minEnclosingCircle(contour)
            center = np.int0(center)
            cv2.circle(image, tuple(center), min(int(r * 0.6), min(shape[0], shape[1]) // 8),
                       np.random.randint(0, 255, 3).tolist(), -1)
        return image

    def do_mosaic(self, image, grayscale_cam, shape):
        """
        马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
        :param frame: opencv frame
        :param int x :  马赛克左顶点
        :param int y:  马赛克右顶点
        :param int w:  马赛克宽
        :param int h:  马赛克高
        :param int neighbor:  马赛克每一块的宽
        """
        image = image.copy()
        frame = image.copy()

        x = 0
        y = 0
        w = shape[1]
        h = shape[1]
        neighbor = min(shape[1], shape[0]) // 40
        fh, fw = frame.shape[0], frame.shape[1]
        if (y + h > fh) or (x + w > fw):
            return
        for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
            for j in range(0, w - neighbor, neighbor):
                rect = [j + x, i + y, neighbor, neighbor]
                color = frame[i + y][j + x].tolist()  # 关键点1 tolist
                left_up = (rect[0], rect[1])
                right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
                cv2.rectangle(frame, left_up, right_down, color, -1)
        image[grayscale_cam > 0.5] = frame[grayscale_cam > 0.5]
        return image

    def show_cam_grad(self, grayscale_cam, src_img):
        """fuse src_img and grayscale_cam and show or save."""

        src_img = np.float32(src_img) / 255

        visualization_img = show_cam_on_image(
            src_img, grayscale_cam, use_rgb=False)
        return visualization_img

    def run(self) -> None:
        while True:
            data, image, grayscale_cams = self.save_queue.get()
            shape = image.shape
            try:
                for label in grayscale_cams:
                    grayscale_cam = grayscale_cams[label]
                    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                    grayscale_cam = cv2.resize(grayscale_cam, (shape[1], shape[0]))

                    heatmap = self.show_cam_grad(grayscale_cam, image)
                    filename = data['image']
                    save_path = '{}_heatmap_{}.jpg'.format(filename, label)
                    cv2.imwrite(save_path, heatmap)

                    mosaic = self.do_mosaic(image, grayscale_cam, shape)
                    paste = self.paste_color_block(image, grayscale_cam, shape)
                    valid_for_image = []
                    # 创建文件夹
                    if mosaic is not None:
                        filename = data['image']
                        save_path = '{}_mosaic_{}.jpg'.format(filename, label)
                        cv2.imwrite(save_path, mosaic)
                        valid_for_image.append(save_path)
                        del mosaic

                    if paste is not None:
                        filename = data['image']
                        save_path = '{}_paste_{}.jpg'.format(filename, label)
                        cv2.imwrite(save_path, paste)
                        valid_for_image.append(save_path)

                        save_path = '{}_transparent_{}.jpg'.format(filename, label)
                        cv2.imwrite(save_path, (0.4 * paste + 0.6 * image).astype(np.uint8))
                        valid_for_image.append(save_path)
                        del paste
                    if len(valid_for_image) > 0:
                        filename = data['image']
                        self.result_queue.put(json.dumps({filename: valid_for_image}, ensure_ascii=False) + '\n')
            except Exception as e:
                print(e)


class Logger(Thread):
    def __init__(self, result_queue: Queue, filename: str):
        super().__init__()
        self.result_queue = result_queue
        self.writer = open(filename, 'w')

    def run(self) -> None:
        while True:
            try:
                log = self.result_queue.get()
                self.writer.write(log)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    args = parse_args()
    model = load_model(args.config, args.device).eval()

    cam_model = Cam(model, args.target_layer, device=args.device)

    image_queue = Queue(10000)
    save_queue = Queue(10000)
    result_queue = Queue(10000)

    num_preprocess_threads = 8
    max_batch_size = 32
    num_postprocess_threrads = 4

    runner = Runner(cam_model, 64, image_queue, save_queue)

    import glob
    import json

    files = open(args.source).readlines()
    files = [json.loads(x.strip()) for x in files]

    chunk_size = len(files) // num_preprocess_threads
    ts = []
    for i in range(num_preprocess_threads):
        loader = Loader(files[i * chunk_size: (i + 1) * chunk_size], image_queue)
        loader.daemon = True
        loader.start()
        ts.append(loader)

    for i in range(num_postprocess_threrads):
        saver = Saver(save_queue, result_queue)
        saver.daemon = True
        saver.start()

    logger = Logger(result_queue, args.source + '.add')
    logger.daemon = True
    logger.start()

    status = sum([int(t.end) for t in ts]) < len(ts)
    while (image_queue.qsize() > 0) or status:
        runner.run()
        status = sum([int(t.end) for t in ts]) < len(ts)

    import time

    while save_queue.qsize() + result_queue.qsize() > 0:
        time.sleep(10)

    import sys

    logger.writer.close()
    sys.exit()
