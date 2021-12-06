import os

import numpy as np
import json

from .multi_label import MultiLabelDataset
from .builder import DATASETS
from mmcls.core import average_performance, mAP
import warnings


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class PornJson(MultiLabelDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = ['normal', 'porn', 'sexy']

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [json.loads(x.strip()) for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for sample in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': sample['image']}
            gt_label = [sample[x] for x in self.CLASSES]
            if len(gt_label) == 1:
                gt_label = gt_label[0]
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 results,
                 metric='mAP',
                 metric_options=None,
                 logger=None,
                 labels=None,
                 **deprecated_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is 'mAP'. Options are 'mAP', 'CP', 'CR', 'CF1',
                'OP', 'OR' and 'OF1'.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'k' and 'thr'. Defaults to None
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            labels (list): label names
            deprecated_kwargs (dict): Used for containing deprecated arguments.

        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'thr': 0.5}

        if deprecated_kwargs != {}:
            warnings.warn('Option arguments for metrics has been changed to '
                          '`metric_options`.')
            metric_options = {**deprecated_kwargs}

        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mAP' in metrics:
            mAP_value = mAP(results, gt_labels)
            eval_results['mAP'] = mAP_value
        if len(set(metrics) - {'mAP'}) != 0:
            performance_keys = ['CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']

            performance_values = average_performance(results, gt_labels, labels=labels,
                                                     **metric_options)
            for k, v in zip(performance_keys, performance_values):
                if k in metrics:
                    eval_results[k] = v
            if labels is not None:
                eval_results.update(performance_values[-1])

        return eval_results
