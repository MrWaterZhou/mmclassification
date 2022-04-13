import cv2
import faiss  # make faiss available
import numpy as np
import onnxruntime


class OnnxModel:
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

    def get_output_name(self):
        """
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self):
        """
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        results = self.onnx_session.run(self.output_name, input_feed=input_feed)
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
        self.features = np.load(feature_path)  # n,d
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
