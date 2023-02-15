import pandas as pd
import pickle
import numpy as np
from lib import ml
from lib import lib as lb
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

""" for img in pd.unique(lb.data.index.get_level_values(0)):
    print(img)
    ml.run_clustering(x=lb.data, img=img, column='NIR', clusteringfun=KMeans)
    ml.run_clustering(x=lb.data, img=img, column='NIR', clusteringfun=SpectralClustering)
    ml.run_clustering(x=lb.data, img=img, column='NIR', clusteringfun=MiniBatchKMeans)
    ml.run_clustering(x=lb.data, img=img, column='NIR', clusteringfun=AgglomerativeClustering)
    ml.run_clustering(x=lb.data, img=img, column='NIR', clusteringfun=Birch) """

""" model = SpectralCoclustering(n_clusters=2, mini_batch=True)
model.fit(np.concatenate(lb.data['NIR_diff'].values).reshape(-1, 1400))
plt.plot(model.column_labels_)
l = model.column_labels_
model = SpectralBiclustering(n_clusters=2, mini_batch=True)
model.fit(np.concatenate(lb.data['NIR_diff'].values).reshape(-1, 1400))
plt.plot(model.column_labels_)
plt.show()

lb.data['c1'], lb.data['c2'] = lb.seperate_bylabel(l, model.column_labels_, lb.data['NIR_255'].values)

for img in pd.unique(lb.data.index.get_level_values(0)):
    print(img)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster1', clusteringfun=KMeans)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster1', clusteringfun=SpectralClustering)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster1', clusteringfun=MiniBatchKMeans)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster1', clusteringfun=AgglomerativeClustering)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster1', clusteringfun=Birch)

input()

for img in pd.unique(lb.data.index.get_level_values(0)):
    print(img)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster2', clusteringfun=KMeans)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster2', clusteringfun=SpectralClustering)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster2', clusteringfun=MiniBatchKMeans)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster2', clusteringfun=AgglomerativeClustering)
    ml.run_clustering(x=lb.data, img=img, time=300, column='cluster2', clusteringfun=Birch) """

""" for img in pd.unique(lb.data.index.get_level_values(0)):
    print(img)
    model = SpectralCoclustering(n_clusters=2, mini_batch=True)
    model.fit(np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400))
    plt.plot(model.column_labels_)
    model = SpectralBiclustering(n_clusters=2, mini_batch=True)
    model.fit(np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400))
    plt.plot(model.column_labels_)
    plt.show() """

""" model = LocalOutlierFactor(n_neighbors=int(len(lb.data)*0.5))
res = model.fit_predict(np.concatenate(lb.data['NIR_minmax'].values).reshape(-1, 1400))
lb.data = lb.data.drop(index=lb.data.iloc[np.where(res == -1)[0]].index) """

#transposed data and binary labels
""" model = LabelBinarizer()
for img in pd.unique(lb.data.index.get_level_values(0)):
    label = np.concatenate(model.fit_transform(lb.data.loc[img].index.get_level_values(0)))
    arr = np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400).T
    for x in arr:
        plt.plot(x, color='grey')
    plt.plot(label, color='red')
    plt.show() """

for img in pd.unique(lb.data.index.get_level_values(0)):
    print(img)
    fig, axis = plt.subplots(1,2)
    for x in lb.data.loc[img]['NIR_255'].values:
        axis[0].plot(x)
    model = SpectralCoclustering(n_clusters=2, mini_batch=True)
    model.fit(np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400))
    axis[0].plot(model.column_labels_, color='red')
    model = SpectralBiclustering(n_clusters=2, mini_batch=True)
    model.fit(np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400))
    axis[0].plot(model.column_labels_, color='green')

    for x in lb.data.loc[img]['NIR_255_smth'].values:
        axis[1].plot(x)
    model = SpectralCoclustering(n_clusters=2, mini_batch=True)
    model.fit(np.concatenate(lb.data.loc[img]['NIR_255_smth'].values).reshape(-1, 1400))
    axis[1].plot(model.column_labels_, color='red')
    model = SpectralBiclustering(n_clusters=2, mini_batch=True)
    model.fit(np.concatenate(lb.data.loc[img]['NIR_255_smth'].values).reshape(-1, 1400))
    axis[1].plot(model.column_labels_, color='green')
    plt.show()