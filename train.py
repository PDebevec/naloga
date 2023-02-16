import pandas as pd
import pickle
import numpy as np
from lib import ml
from lib import lib as lb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, TruncatedSVD


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

#rezultati z decomposition za NIR_diff in NIR_minmax
""" csv = pd.read_csv('allrez.csv').set_index(['col', 'component_fun', 'setting', 'model', 'video'])
arr = np.array([])
for img in pd.unique(csv.index.get_level_values(4)):
    temp = csv.xs(img, level='video', drop_level=False)
    i = np.argwhere( temp.values == np.max(temp.values) )[:, 0].flatten()
    print(temp.iloc[ i ])
    arr = np.concatenate((temp.iloc[ i ].values[:, 0], arr))
print(np.average(arr), np.min(arr), np.max(arr)) """

#join
csvminmax = pd.read_csv('rez1.csv')
csvminmax['col'] = 'NIR_minmax'
csvdiff = pd.read_csv('rez.csv')
csvdiff['col'] = 'NIR_diff'
csv =  pd.concat([ csvminmax, csvdiff ]).set_index([ 'col', 'component_fun', 'setting', 'model', 'video' ])
csv.to_csv('allrez.csv')

#decomposition za NIR_diff in NIR_minmax
""" csv = pd.read_csv('rez1.csv').set_index(['component_fun', 'setting', 'model', 'video'])
dfcsv = np.array(['component_fun', 'setting',  'model', 'video', 'acc'])
for img in pd.unique(lb.data.index.get_level_values(0)):
    #print(img)
    model = TruncatedSVD(n_components=12)#, kernel='cosine')
    x = model.fit_transform(np.concatenate(lb.data.loc[img]['NIR_minmax'].values).reshape(-1 ,1400))
    
    compmodel = type(model).__name__
    
    model = KMeans(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'default', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = SpectralClustering(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'default', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = MiniBatchKMeans(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'default', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'default', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))
    model = Birch(n_clusters=2)
    model.fit(x)
    dfcsv = np.vstack((dfcsv, np.array([compmodel, 'default', type(model).__name__, img, ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))])))

dfcsv = pd.DataFrame(dfcsv[1:], columns=dfcsv[0])
dfcsv = dfcsv.sort_values(by=['component_fun', 'setting', 'model', 'video']).set_index(['component_fun', 'setting', 'model', 'video'])
csv = pd.concat([csv, dfcsv])
csv.to_csv('rez1.csv') """

#transposed data and binary labels
""" model = LabelBinarizer()
for img in pd.unique(lb.data.index.get_level_values(0)):
    label = np.concatenate(model.fit_transform(lb.data.loc[img].index.get_level_values(0)))
    arr = np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400).T
    for x in arr:
        plt.plot(x, color='grey')
    plt.plot(label, color='red')
    plt.show() """

#co in bi clustering
""" for img in pd.unique(lb.data.index.get_level_values(0)):
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
    plt.show() """

#>90% acc
""" model = KNeighborsClassifier()
split = 220
col = 'NIR_diff'
best = bres = 0
for i in range(100):
    x_train = lb.data.xs('Cancer', level=1, drop_level=False)[col].sample(frac=1)
    x_test, x_train = x_train[split:], x_train[:split]
    temp = lb.data.xs('Benign', level=1, drop_level=False)[col].sample(frac=1)
    x_test, x_train = pd.concat([ x_test, temp[split:] ]), pd.concat([ x_train, temp[:split] ])
    temp = lb.data.xs('Healthy', level=1, drop_level=False)[col].sample(frac=1)
    x_test, x_train = pd.concat([ x_test, temp[split:] ]).to_frame().sample(frac=1), pd.concat([ x_train, temp[:split] ]).to_frame().sample(frac=1)
    model.fit(np.concatenate(x_train[col].values).reshape(-1, 1400), x_train.index.get_level_values(1))

    res = model.predict(np.concatenate(x_test[col].values).reshape(-1, 1400))

    count = 0
    for i in range(len(res)):
        if res[i] == x_test.index.get_level_values(1)[i]:
            count+=1
    #print(len(res), count, (count / len(res)))
    if best < (count / len(res)):
        best = (count / len(res))
        bres = res
print(best)
#print(np.concatenate((np.array(x_test.index.get_level_values(1))[..., np.newaxis], res[..., np.newaxis]), axis=1)) """