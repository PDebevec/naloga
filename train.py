import pandas as pd
import pickle
import numpy as np
import ml
import lib as lb
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

#double decomposition in clustering
#izpis acc kombinacij pri img s slabšimi acc
""" csv = pd.read_csv('all.csv').set_index( ['col', 'img', 'decomposition.1',  'decomposition.2', 'cluster'] )
videos = []
algos = []
for img in pd.unique(lb.data.index.get_level_values(0)):
    x = csv.xs(img, level='img')
    y = np.array(x['acc'].tolist())
    if x.iloc[y.astype(float).argsort()[-1]]['acc'] == 1:
        videos.append(img)
    else:
        algos.append(x.reset_index().iloc[y.astype(float).argsort()[-1]].values)

for img in videos:
    print(img)
    x = csv.xs(img, level='img')
    for algo in algos:
        print(algo[0], algo[1], algo[2], algo[3], end=' ')
        print(x.loc[algo[0], algo[1], algo[2], algo[3]]['acc']) """
#
csv = pd.read_csv('all.csv').set_index( ['col', 'img', 'decomposition.1',  'decomposition.2', 'cluster'] )
comb = []
for img in pd.unique(csv.index.get_level_values(1)):
    temp = csv.loc[:, img, :, :, :]
    mx = np.max(temp['acc'].values)
    t = temp.index[temp['acc'] == mx].tolist()
    comb.append(t)
countcomb = []
comb = [' '.join(x) for i in comb for x in i]
u, counts = np.unique(comb, return_counts=True)
r = np.hstack(( np.array(u).reshape(-1,1), np.array(counts).reshape(-1,1) ))
br = r[np.where(r[:, 1] == '7')]
csv = csv.reset_index(level='img').sort_index()
for x in br:
    temp = csv.loc[x[0].split(' ')[0], x[0].split(' ')[1], x[0].split(' ')[2], x[0].split(' ')[3]]
    print(temp.index.tolist()[0])
    print(np.mean(temp['acc'].values), np.min(temp['acc'].values))

#random forest na decomposition
#1.
""" x_train, x_test = ml.seperate_x_random_img()
#x_train, x_test = ml.seperate_x_cancer_benign(0.1)
for col in lb.data.columns[10:]:
    model = RandomForestClassifier()
    model.fit(np.concatenate(x_train[col].values).reshape(-1, 12), x_train.index.get_level_values(1))
    res = model.predict(np.concatenate(x_test[col].values).reshape(-1, 12))

    count = 0
    for i in range(len(res)):
        if res[i] == x_test.index.get_level_values(1)[i]:
            count+=1
    print(col, count, len(res), count/len(res)) """
#2.
""" x_train, x_test = ml.seperate_x_random_img()

forest = RandomForestClassifier()
forest.fit(np.array(x_train['NIR_diff_smth_FastICA'].tolist(), dtype=float), x_train.index.get_level_values(1))

model = AgglomerativeClustering(n_clusters=2)
cluster = model.fit(np.array(x_test['NIR_diff_smth_FastICA'].tolist(), dtype=float))

labels = forest.predict(np.array(x_test['NIR_diff_smth_FastICA'].tolist(), dtype=float))

i0 = np.where(cluster.labels_ == 0)[0]
i1 = np.where(cluster.labels_ == 1)[0]

print(np.unique(labels[i0], return_counts=True))
print(np.unique(labels[i1], return_counts=True))
print(np.unique(labels, return_counts=True))
print(pd.unique(x_test.index.get_level_values(1))) """

#decomposition in cluster podatkov
#naredi
""" ml.decomposition_cluster('NIR_255')
ml.decomposition_cluster('NIR_diff')
ml.decomposition_cluster('NIR_minmax')
ml.decomposition_cluster('NIR_255_smth')
ml.decomposition_cluster('NIR_diff_smth')
ml.decomposition_cluster('NIR_minmax_smth') """
#skupi
""" csv255s = pd.read_csv('NIR_255_smth.csv')
csvdiffs = pd.read_csv('NIR_diff_smth.csv')
csvminmaxs = pd.read_csv('NIR_minmax_smth.csv')
csv255 = pd.read_csv('NIR_255.csv')
csvdiff = pd.read_csv('NIR_diff.csv')
csvminmax = pd.read_csv('NIR_minmax.csv')
csv =  pd.concat([ csvdiff, csvminmax, csv255, csvdiffs, csvminmaxs, csv255s ]).set_index([ 'col', 'component_fun', 'setting', 'model', 'video' ])
csv.to_csv('allrez.csv') """
#izpiše najbolše
""" csv = pd.read_csv('allrez.csv').set_index(['col', 'component_fun', 'setting', 'model', 'video'])
arr = np.array([])
for img in pd.unique(csv.index.get_level_values(4)):
    temp = csv.xs(img, level='video', drop_level=False)
    i = np.argwhere( temp.values[:, 0] == np.max(temp.values[:, 0]) )[:, 0].flatten()
    temp = temp.iloc[ i ]
    i = np.argwhere( (temp.values[:, 1]).astype(int) == np.min((temp.values[:, 1]).astype(int)) )[:, 0].flatten()
    print(temp.iloc[ i ])
    arr = np.concatenate((temp.iloc[ i ].values[:, 0], arr))
print('avg: ', np.average(arr), '\nmin:', np.min(arr), '\nmax:', np.max(arr)) """
#najbolša kobinacija train.txt
""" csv = pd.read_csv('allrez.csv').set_index(['col', 'component_fun', 'setting', 'model', 'video']).sort_index()

csv = csv.xs('NIR_255', level='col', drop_level=False)
arr = []
for l0 in pd.unique(csv.index.get_level_values(0)):
    temp = csv.xs(l0, level=0, drop_level=False)
    for l1 in pd.unique(temp.index.get_level_values(1)):
        l1t = temp.xs(l1, level=1, drop_level=False)
        for l2 in pd.unique(l1t.index.get_level_values(2)):
            l2t = l1t.xs(l2, level=2, drop_level=False)
            for l3 in pd.unique(l2t.index.get_level_values(3)):
                l3t = l2t.xs(l3, level=3, drop_level=False)
                arr.append([
                    l0+' '+l1+' '+l2+' '+l3
                    ,np.mean(l3t['acc'].values)
                    ,np.min(l3t['acc'].values)
                    #,l3t['acc'].values.tolist().count(1)
                    ])
arr = np.array(arr)
temp = []
for comb in arr[:, 0]:
    temp.append(
        np.where(arr[arr[:, 1].astype(float).argsort()][:, 0] == comb)[0][0]
        + np.where(arr[arr[:, 2].astype(float).argsort()][:, 0] == comb)[0][0]
        #+ np.where(arr[arr[:, 3].astype(int).argsort()][:, 0] == comb)[0][0]
    )
#print(np.array(temp))
arr = np.hstack((arr, np.array(temp).reshape(-1, 1)))
print(arr[arr[:, 3].astype(float).argsort()][-1])
#print(csv.loc['NIR_diff_smth', 'FastICA', 'default', 'AgglomerativeClustering']) """

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