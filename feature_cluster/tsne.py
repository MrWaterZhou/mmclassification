from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json


def tsne(data: np.ndarray, labels: list):
    tsne = TSNE(n_components=2, )
    X_tsne = tsne.fit_transform(data)
    X_tsne_data = np.vstack((X_tsne.T, labels)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'label'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='label', x='Dim1', y='Dim2')
    plt.savefig('x.jpg', dpi=100)


def kmeans(data: np.ndarray, labels):
    km = KMeans(n_clusters=8,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    pca = PCA(64)
    data_pca = pca.fit_transform(data)
    km.fit(data_pca)
    # 将原始数据中的索引设置成得到的数据类别
    X_out = pd.DataFrame(data_pca, index=km.labels_)
    X_out_center = pd.DataFrame(km.cluster_centers_)  # 找出聚类中心
    # 将中心放入到数据中，一并tsne，不能单独tsne
    X_outwithcenter = X_out.append(X_out_center)

    # 用TSNE进行数据降维并展示聚类结果
    from sklearn.manifold import TSNE
    tsne = TSNE()
    tsne.fit_transform(X_outwithcenter)  # 进行数据降维,并返回结果
    X_tsne = pd.DataFrame(tsne.embedding_)
    X_tsne_data = np.vstack((X_tsne.T, labels)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'label'])
    # 将index化成原本的数据的index，tsne后index会变化
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='label', x='Dim1', y='Dim2')
    plt.savefig('x.jpg', dpi=100)

    return km.labels_


if __name__ == '__main__':
    features = np.load('train.npy')
    labels = [json.loads(x.strip()) for x in open('train')]
    idx = [i for i, x in enumerate(labels) if x['性感_胸部'] == 1]
    s_feature = features[idx]
    s_data = ['sexy_breast' for i in idx]

    cluster_labels = kmeans(s_feature, s_data)
    with open('x.txt', 'w') as f:
        for l, i in zip(cluster_labels, idx):
            tmp = labels[i]
            tmp['cluster_label'] = int(l)
            f.write(json.dumps(tmp, ensure_ascii=False) + '\n')
