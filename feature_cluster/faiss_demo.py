import cv2
import faiss  # make faiss available
import numpy as np
import json
from eval import Model


class OnnxModel:
    def __init__(self, engine_path, max_batch_size=64):
        self.model = Model(engine_path, max_batch_size, 224, 224)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

    def forward(self, image_numpy):
        '''
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        results = self.model.infer(image_numpy)
        return results

    def get_normalized_feature(self, image_path_list: list):
        if isinstance(image_path_list, str):
            image_path_list = [image_path_list]
        images = []
        for image_path in image_path_list:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, 0)
            images.append(image)
        image = np.concatenate(images, 0)
        image = (image - self.mean) / self.std
        image = np.transpose(image, (0, 3, 1, 2))
        image = np.ascontiguousarray(image, dtype=np.float32)
        feature = self.forward(image)[0]  # b, 1360
        feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        return feature

    def get_most_similar(self, image_path, image_path_list):
        f1 = self.get_normalized_feature(image_path)  # 1,1360
        if isinstance(image_path_list, list):
            f2 = self.get_normalized_feature(image_path_list)  # b,1360
        else:
            image_path_list, f2 = image_path_list
        scores = np.dot(f2, f1.T)[:, 0].tolist()
        scores = zip(scores, image_path_list)
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        return scores[0]


class FaissSearch:
    def __init__(self, feature_path: str, filenames: list, model: OnnxModel):
        if isinstance(feature_path, str):
            self.features = np.load(feature_path)  # n,d
        else:
            self.features = feature_path
        self.filenames = filenames
        d = self.features.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.features)
        self.model = model

    def find_n_nearest(self, filename_list: list, show=True):
        if isinstance(filename_list, str):
            filename_list = [filename_list]
        features = self.model.get_normalized_feature(filename_list)
        scores, idxes = self.index.search(features, 5)
        for fname, score, idx in zip(filename_list, scores, idxes):
            neighbors = [self.filenames[i] for i in idx]
            for s, n in zip(score, neighbors):
                print(fname, s, n)

    def find_by_score(self, filename: str):
        filename_list = [filename]
        features = self.model.get_normalized_feature(filename_list)
        _, scores, idxes = self.index.range_search(features, 0.20)
        neighbors = [self.filenames[i] for i in idxes]
        for s, n in zip(scores, neighbors):
            print(filename, s, n)
        return neighbors


def try_demo():
    model = OnnxModel('/data/xialang/projects/mmclassification/work_dirs/porn_20220404_multiCenter/feature.engine')
    data = [json.loads(x.strip()) for x in open('train')]
    searcher = FaissSearch('train.npy', data, model)
    return searcher


def find_and_extend(filename: str, save_path='result.txt'):
    searcher = try_demo()
    results = []
    results.extend(searcher.find_by_score(filename))

    filenames = [x['image'] for x in results if x['性感_胸部'] == 0]
    for filename in filenames:
        results.extend(searcher.find_by_score(filename))

    uniq = set()
    with open(save_path, 'w') as f:
        for x in results:
            if x['image'] not in uniq:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
                uniq.add(x['image'])


def find_and_extend_clean(filenames: list, save_path='result.txt'):
    searcher = try_demo()
    results = []

    for filename in filenames:
        results.extend(searcher.find_by_score(filename))

    uniq = set()
    with open(save_path, 'w') as f:
        for x in results:
            if x['image'] not in uniq:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
                uniq.add(x['image'])


def find_similar():
    model = OnnxModel('/data/xialang/projects/mmclassification/work_dirs/porn_20220404_multiCenter/feature.engine')
    data = [json.loads(x.strip()) for x in open('train')]
    features = np.load('train.npy')
    filenames = [x.strip() for x in open('/data/xialang/tmp_data/轻度性感_胸部.txt')]
    filename_set = set(filenames)
    idx = [i for i, x in enumerate(data) if x['image'] in filename_set]
    cluster_feature = features[idx]
    searcher = FaissSearch(cluster_feature, filenames, model)

    with open('轻度性感_胸部_v2.txt', 'w') as f:
        for x in data:
            if x['image'] not in filename_set:
                n = searcher.find_by_score(x['image'])
                if len(n) > 5:
                    f.write(json.dumps(x, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    labels = ['性感_胸部',
              '色情_女胸',
              '性感_腿部特写',
              '色情_男下体',
              '色情_口交',
              '性感_内衣裤',
              '性感_男性胸部',
              '色情_裸露下体']
    data = [json.loads(x.strip()) for x in open('../data/porn/train_v4.txt')]
    model = OnnxModel('model.onnx')
    searcher = FaissSearch('train.txt.features.npy', data, model)
    with open('knn_result.txt', 'w') as knn_result:
        batch_size = 10
        start = 0
        while start < len(data):
            scores, idxes = searcher.index.search(searcher.features[start:start + batch_size], 10)
            for source_data, score, idx in zip(data[start:start + batch_size], scores, idxes):
                neighbors = [data[i] for i, s in zip(idx[2:], score[2:]) if s < 0.15]
                neighbors_result = {x: 0 for x in labels}
                for label in labels:
                    for n in neighbors:
                        neighbors_result[label] += n[label]
                    neighbors_result[label] = 1 if neighbors_result[label] > 1 else 0
                    if neighbors_result[label] == source_data[label]:
                        neighbors_result.pop(label)
                neighbors_result['image'] = source_data['image']
                neighbors_result['neighbor_images'] = [n['image'] for n in neighbors]
                if len(neighbors_result) > 2:
                    knn_result.write(json.dumps(neighbors_result, ensure_ascii=False) + '\n')
            start += batch_size
