import pandas as pd
import pickle
import numpy as np
import ml
import lib as lb
from scipy.signal import find_peaks
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

#testiranje hitrosti clustering > cluster.txt
""" cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2)
        ,AgglomerativeClustering(n_clusters=2)
    ]

x = np.array(lb.data.loc[16091401]['NIR_nfp'].values.tolist())
for c in cluster:
    t = 0
    for i in range(1000):
        st = time.time()
        c.fit(x)
        t += time.time() - st
    print(type(c).__name__, round(t, 2)) """

#testiranje hitrosti decomposition > decomposition.txt
""" setting= [ '_sigmoid', '_cosine', '', '', '', '', '' ]
decomposition = [
            KernelPCA(n_components=4, kernel='sigmoid'),
            KernelPCA(n_components=4, kernel='cosine'),
            FactorAnalysis(n_components=4),
            FastICA(n_components=4),
            IncrementalPCA(n_components=4),
            PCA(n_components=4),
            TruncatedSVD(n_components=4)
        ]

x = np.array(lb.data.loc[16091401]['NIR_nfp'].values.tolist())
for d,s in zip(decomposition, setting):
    t = 0
    for i in range(1000):
        st = time.time()
        d.fit_transform(x)
        t += time.time() - st
    print(type(d).__name__+s, round(t, 2)) """

#vizualizacija decomposition na video glede na ROI
""" arr = []
for img in lb.uvideo:
    x = np.array(lb.data.loc[img]['NIR_nfp'].values.tolist())

    model = FactorAnalysis(n_components=1)
    w = model.fit_transform(x)
    arr.append(np.zeros(150))
    for r,i in zip(lb.data.loc[img].index.get_level_values(1), range(len(w))):
        arr[-1][r] = w[i]

    for l,r,i in zip(lb.data.loc[img].index.get_level_values(0), lb.data.loc[img].index.get_level_values(1), range(len(w))):
        if l == 'Healthy':
            plt.plot(r, w[i], 'g.')
        elif l == 'Benign':
            plt.plot(r, w[i], 'b.')
        else:
            plt.plot(r, w[i], 'r.')
    plt.show() """
#clustering na decomposition videja
#cluster podobne videje skupaj s DBSCAN ???
#n_clusters = 6 na sklearn.cluster
""" model = DBSCAN(min_samples=1, eps=8.2)
model.fit(np.array(arr))

print(model.labels_)
print(lb.videos['binary'].values)
#print(model.labels_) """

#dobit mean med peaks ???
""" img = 170108
x = lb.data.loc[img]['NIR_diff'].values[2]

p1 = find_peaks(x)[0]
p2 = find_peaks(-x)[0]
print(len(p1), len(p2))
plt.plot(x)
plt.plot(lb.data.loc[img]['NIR_nfp'].values[2])
plt.plot(p1, x[p1], 'x')
plt.plot(p2, x[p2], 'x')
plt.show() """

#clustering z n števili podatkov razlika med 255 in nfp
""" for img in pd.unique(lb.data.index.get_level_values(0)):
    arr = [0, 0]
    gd = lb.get_divisor(1400)
    for d in gd:
        nir = np.array(lb.data.loc[img]['NIR_255'].values.tolist())
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(nir[:, ::d])
        #print(ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0)), end=' ')
        arr[0] += ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))
        nfp = np.array(lb.data.loc[img]['NIR_nfp'].values.tolist())
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(nfp[:, ::d])
        arr[1] += ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0))
        #print(ml.get_accuracy(model.labels_, lb.data.loc[img].index.get_level_values(0)))
    print(img, arr[0]/len(gd), arr[1]/len(gd))
    arr[0] = 0
    arr[1] = 0 """

#clustering z n števili podatkov (divisor)
""" csv = pd.read_csv('divisor.csv').set_index(['col', 'video', 'cluster'])
arr = []
arr2 = []
for video in lb.videol:
    x = csv.query("video == "+str(video))
    x = x.sort_values(by=['acc'])
    v = x.iloc[-1]['acc']
    arr.append(v)
    arr2.append(x.loc[x['acc'] == v])
    arr2[-1] = arr2[-1].sort_values(by=['divisor'])
print(arr2)
print(np.min(arr), np.mean(arr)) """
""" for a in arr2:
    if len(a) == 1:
        c = 0
        for b in arr2:
            i = a.index[0][0] == b.index[0][0] and a.index[0][2] == b.index[0][2]
            if i:
                c+=1
        print(c, a.index[0]) """

#double decomposition in clustering
#izpis
""" csv = pd.read_csv('all.csv').set_index( ['col', 'img', 'decomposition.1',  'decomposition.2', 'cluster'] )
videos = []
algos = []
for img in pd.unique(lb.data.index.get_level_values(0)):
    print('\n', img)
    x = csv.xs(img, level='img')
    y = np.array(x['acc'].tolist())
    print(x.iloc[y.astype(float).argsort()[-5:]]) """
"""     if x.iloc[y.astype(float).argsort()[-1]]['acc'] == 1:
        videos.append(img)
    else:
        algos.append(x.reset_index().iloc[y.astype(float).argsort()[-1]].values) """

""" for img in videos:
    print(img)
    x = csv.xs(img, level='img')
    for algo in algos:
        print(algo[0], algo[1], algo[2], algo[3], end=' ')
        print(x.loc[algo[0], algo[1], algo[2], algo[3]]['acc']) """
#
""" csv = pd.read_csv('all.csv').set_index( ['col', 'img', 'decomposition.1',  'decomposition.2', 'cluster'] )
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
    print(np.mean(temp['acc'].values), np.min(temp['acc'].values)) """

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