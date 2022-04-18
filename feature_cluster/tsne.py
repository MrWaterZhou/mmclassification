from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import json


def tsne(data: np.ndarray, labels: list):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(data)
    X_tsne_data = np.vstack((X_tsne.T, labels)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'label'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, hue='label', x='Dim1', y='Dim2')
    plt.imsave('x.png')


if __name__ == '__main__':
    features = np.load('train.npy')
    labels = [json.loads(x.strip()) for x in open('train')]
    idx = [i for i, x in enumerate(labels) if x['性感_胸部'] == 1]
    s_feature = features[idx]
    s_data = ['性感_胸部' for i in idx]

    tsne(s_feature,s_data)