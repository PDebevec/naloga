#import sklearn as sk
import numpy as np
import pickle
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
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

for img in pd.unique(lb.data.index.get_level_values(0)):
    for label in pd.unique(lb.data.loc[img].index.get_level_values(0)):
        c = ''
        match label:
            case 'Healthy': c = 'green'
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        for x in lb.data.loc[img, label]['NIR_255']:
            plt.plot(x, color=c)
    plt.savefig('graphs/255/'+str(img)+'255.png')
    #plt.savefig('graphs/'+str(img)+'.png')
    plt.show()

""" model = SpectralCoclustering(n_clusters=2)
model.fit(np.concatenate(lb.data.loc[16093501]['NIR_diff'].values).reshape(-1, 1400))
print(model.row_labels_, model.column_labels_)
model = SpectralBiclustering(n_clusters=2)
model.fit(np.concatenate(lb.data.loc[16093501]['NIR_diff'].values).reshape(-1, 1400))
print(model.row_labels_, model.column_labels_) """

""" model = LocalOutlierFactor(n_neighbors=int(len(lb.data)*0.5))
res = model.fit_predict(np.concatenate(lb.data['NIR_minmax'].values).reshape(-1, 1400))
lb.data = lb.data.drop(index=lb.data.iloc[np.where(res == -1)[0]].index) """


""" for img in pd.unique(lb.data.xs('Benign' and 'Healthy', level='finding', drop_level=False).index.get_level_values(0)):
    print(img)
    ml.run_clustering(x=lb.data, img=img, time=300, column='NIR_diff', clusteringfun=KMeans)
    ml.run_clustering(x=lb.data, img=img, time=300, column='NIR_diff', clusteringfun=SpectralClustering)
    ml.run_clustering(x=lb.data, img=img, time=300, column='NIR_diff', clusteringfun=MiniBatchKMeans)
    ml.run_clustering(x=lb.data, img=img, time=300, column='NIR_diff', clusteringfun=AgglomerativeClustering)
    ml.run_clustering(x=lb.data, img=img, time=300, column='NIR_diff', clusteringfun=Birch) """

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