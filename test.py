#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dask.dataframe as df
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.sparse as sparse
from lib import lib as lb
from lib import ml
#import lib2 as lb2

#data['label'] = lb.get_lables(data)
#print(lb.data.xs('Cancer', level='finding', drop_level=False))
lb.data['NIR_minmax'] = lb.get_normalized(lb.data['NIR'])
lb.data['NIR_minmax_img'] = lb.get_normalized_byimg(lb.data['NIR'])
lb.data['NIR_255'] = lb.data['NIR']/255
lb.data['NIR_diff'] = lb.get_diff(lb.data['NIR'])
#16091401 170108

csv = pd.DataFrame(columns=['normalized', 'algo', 'acc']).set_index('normalized')

#print(ml.run_clustering(x=lb.data, img=16091401, column='NIR', clusteringfun=KMeans))
csv.loc['NIR'] = [ ['ne', 123], ['ja', 321] ]

csv.loc['NIR'] = [  ml.run_clustering(x=lb.data, img=16091401, column='NIR', clusteringfun=KMeans), 
                    ml.run_clustering(x=lb.data, img=16091401, column='NIR', clusteringfun=SpectralClustering),
                    ml.run_clustering(x=lb.data, img=16091401, column='NIR', clusteringfun=MiniBatchKMeans),
                    ml.run_clustering(x=lb.data, img=16091401, column='NIR', clusteringfun=AgglomerativeClustering),
                    ml.run_clustering(x=lb.data, img=16091401, column='NIR', clusteringfun=Birch) ]

print(csv)

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

""" for img in pd.unique(lb.data.index.get_level_values(0)):
    for label in pd.unique(lb.data.loc[img].index.get_level_values(0)):
        c = ''
        match label:
            case 'Healthy': c = 'green'
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        for x in lb.data.loc[img, label]['NIR_diff']:
            plt.plot(x, color=c)
    #plt.savefig('graphs/minmax/'+str(img)+'_minmax.png')
    #plt.savefig('graphs/'+str(img)+'.png')
    plt.show() """

#exit()