import time
import pickle
import lib as lb
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor


def run_clustering(x, img, clusteringfun, column):
    x = x.loc[img]
    model = clusteringfun(n_clusters=2)
    model.fit(np.concatenate(x[column].values).reshape(-1, len(x[column][0])))
    return get_accuracy(model.labels_, x.index.get_level_values(2))


def get_accuracy(labels_, y):
    #model = LabelBinarizer()
    #res = model.fit_transform(y).reshape(-1)
    acc = accuracy_score(y, labels_)
    return max([ acc, (acc-1)*-1 ])

def get_f1_score(labels_, y, ave='micro'):
    #model = LabelBinarizer()
    #res = model.fit_transform(y).reshape(-1)
    acc = f1_score(y, labels_, average=ave)
    return max([ acc, (acc-1)*-1 ])

def separate_labels(labels_, y_train):
    arr = []
    for label in np.unique(y_train):
        arr.append(list(labels_[np.where(y_train == label)]))
    separate_labels = arr
    for x in np.unique(np.concatenate(separate_labels)):
        arr[np.argmax([ y.count(x) for y in separate_labels])].append(x)
    return arr


def find_batch_inlabel(separate_labels, labels_, ulabels):
    arr = []
    for x in labels_:
        arr.append(ulabels[np.argmax([ y.count(x) for y in separate_labels ])])
    return arr


def find_batch(labels_, labeled, labels):
    arr = []
    for batch in labels_:
        arr.append(labels[np.argmax([x.count(batch) for x in labeled])])
    return arr

def select_decomposition_cluster(decomposition, cluster, column, x_fit):
    for img in pd.unique(x_fit.index.get_level_values(0)):
            x = decomposition.fit_transform(x_fit.loc[img][column])
            cluster.fit(x)
    return cluster.lables_

def get_cancer_benign():
    cvideos = pd.unique(lb.data.xs('Cancer', level=1, drop_level=False).index.get_level_values(0))
    bvideos = pd.unique(lb.data.xs('Benign', level=1, drop_level=False).index.get_level_values(0))
    return cvideos, bvideos

def seperate_x_cancer_benign(split):
    c, b = get_cancer_benign()
    c_train, c_test = train_test_split(c, shuffle=True, test_size=split)
    b_train, b_test = train_test_split(b, shuffle=True, test_size=split)
    return lb.data.loc[np.hstack((c_train, b_train))], lb.data.loc[np.hstack((c_test, b_test))]

def seperate_x_random_img():
    img = np.random.choice(pd.unique(lb.data.index.get_level_values(0)), 1)[0]
    return lb.data.drop(img), lb.data.xs(img, level='video', drop_level=False)

def decomposition_data(column, file, components):
    setting= [ '_sigmoid_', '_cosine_', '_', '_', '_', '_', '_' ]

    decomposition = [
        KernelPCA(n_components=components, kernel='sigmoid'),
        KernelPCA(n_components=components, kernel='cosine'),
        FactorAnalysis(n_components=components),
        FastICA(n_components=components),
        IncrementalPCA(n_components=components),
        PCA(n_components=components),
        TruncatedSVD(n_components=components)
    ]
    
    nplist = []
    for s,d in zip(setting, decomposition):
        nplist.append(np.array([[column+s+type(d).__name__]*components]))
        for img in pd.unique(file.index.get_level_values(0)):
            nplist[-1] = np.vstack(( nplist[-1],
            d.fit_transform(lb.get_x(img, column))))
        file[column+s+type(d).__name__] = nplist[-1][1:].tolist()
    return file

def decomposition_cluster(column):
    setting= [ 'sigmoid', 'cosine', 'default', 'default', 'default', 'default', 'default' ]

    decomposition = [
        KernelPCA(n_components=12, kernel='sigmoid'),
        KernelPCA(n_components=12, kernel='cosine'),
        FactorAnalysis(n_components=12),
        FastICA(n_components=12),
        IncrementalPCA(n_components=12),
        PCA(n_components=12),
        TruncatedSVD(n_components=12)
    ]

    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2)
        ,AgglomerativeClustering(n_clusters=2)
    ]
    
    dfcsv = np.array(['col', 'component_fun', 'setting',  'model', 'video', 'acc', 'time'])
    for img in pd.unique(lb.data.index.get_level_values(0)):
        for d, s in zip(decomposition, setting):
            sd = time.time()
            x = d.fit_transform(lb.get_x(img, column))
            dt = time.time() - sd
            compmodel = type(d).__name__
            for c in cluster:
                sc = time.time()
                c.fit(x)
                ct = time.time() - sc
                dfcsv = np.vstack((dfcsv, np.array([column, compmodel, s, type(c).__name__, img, get_accuracy(c.labels_, lb.data.loc[img].index.get_level_values(2)), (dt+ct)*1000])))
    dfcsv = pd.DataFrame(dfcsv[1:], columns=dfcsv[0])
    dfcsv = dfcsv.sort_values(by=['col', 'component_fun', 'setting', 'model', 'video']).set_index(['col', 'component_fun', 'setting', 'model', 'video'])
    Path(column+'.csv').touch(exist_ok=True)
    dfcsv.to_csv(column+'.csv')
    print(column+'.csv')
    return

def double_decomposition_cluster():
    setting= [ '_sigmoid', '_cosine', '', '', '', '', '' ]

    decomposition = [
            KernelPCA(n_components=7, kernel='sigmoid'),
            KernelPCA(n_components=7, kernel='cosine'),
            FactorAnalysis(n_components=7),
            FastICA(n_components=7),
            IncrementalPCA(n_components=7),
            PCA(n_components=7),
            TruncatedSVD(n_components=7)
        ]

    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2)
        ,AgglomerativeClustering(n_clusters=2)
    ]

    arr = []
    for col in lb.data.columns[2:8]:
        for img in pd.unique(lb.data.index.get_level_values(0)):
            x = np.array(lb.data.loc[img][col].tolist())
            for s1,d1 in zip(setting, decomposition):
                for s2,d2 in zip(setting, decomposition):
                    ds = time.time()
                    w = d1.fit_transform(x)
                    h = d2.fit_transform(w.T)
                    de = time.time() - ds
                    for c in cluster:
                        print(col, img, type(d1).__name__, type(d2).__name__, type(c).__name__)
                        cs = time.time()
                        c.fit(np.matmul(w, h))
                        ce = time.time() - cs
                        arr.append([
                            col, img,
                            type(d1).__name__+s1, type(d2).__name__+s2,
                            type(c).__name__,
                            get_accuracy(c.labels_, lb.data.loc[img].index.get_level_values(2)),
                            (de+ce)*1000
                            ])
            return arr
    df = pd.DataFrame(arr, columns=['col', 'img', 'decomposition.1',  'decomposition.2', 'cluster', 'acc', 'time']).set_index( [ 'col', 'img', 'decomposition.1',  'decomposition.2', 'cluster' ] )
    df.to_csv('aalldiff.csv')
    return

def divisor():

    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2)
        ,AgglomerativeClustering(n_clusters=2)
    ]

    arr = []
    i = 0
    for col in lb.data.columns[1:]:
        divisors = lb.get_divisor(len(lb.data[col].values[0]))
        for img in pd.unique(lb.data.index.get_level_values(0)):
            print(col, img)
            for c in cluster:
                for d in divisors:
                    x = lb.data.loc[img][col]
                    st = time.time()
                    c.fit(np.array(x.values.tolist())[:, ::d])
                    et = time.time() - st
                    arr.append([
                        col, img,
                        type(c).__name__, d,
                        get_accuracy(c.labels_, x.index.get_level_values(2)),
                        et*1000
                    ])
    df = pd.DataFrame(arr, columns=['col', 'video', 'cluster', 'divisor', 'acc', 'time']).set_index(['col', 'video', 'cluster', 'divisor'])
    df.to_csv('divisor.csv')
    return

def clustering_on_column(column):
    open('./csv/'+column+'.csv', 'w').close()

    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2, threshold=0.05)
        ,AgglomerativeClustering(n_clusters=2)
    ]

    f = open('./csv/'+column+'.csv', 'a')
    f.write('video,clustering,acc,time\n')
    for img in pd.unique(lb.data.index.get_level_values(0)):
        x = lb.data.loc[img][column]
        for c in cluster:
            st = time.time()
            c.fit(np.array(x.values.tolist()))
            et = time.time() - st
            #print(column, img, type(c).__name__, get_accuracy(c.labels_, x.index.get_level_values(0)), et*1000, sep=',')
            f.write(str(img)+','+type(c).__name__+','+str(get_accuracy(c.labels_, x.index.get_level_values(2)-1))+','+str(et*1000)+'\n')
    return

""" def clustering_on_column_outlier(column):
    open('./csv/'+column+'+outlier.csv', 'w').close()
    
    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2)
        ,AgglomerativeClustering(n_clusters=2)
    ]

    f = open('./csv/'+column+'+outlier.csv', 'a')
    f.write('video,clustering,acc,time\n')
    for img in lb.uvideo:
        x = lb.get_x(img, column)
        model = LocalOutlierFactor(n_neighbors=int(len(x)*0.9))
        res = model.fit_predict(x)
        
        for c in cluster:
            st = time.time()
            c.fit(x[res == 1])
            et = time.time() - st
            #print(img, get_accuracy(model.labels_, lb.get_l(img, l=2)[res == 1]))
            f.write(str(img)+','+type(c).__name__+','+str(get_accuracy(c.labels_, x.index.get_level_values(2)))+','+str(et*1000)+'\n')
    return """

def clustering_on_diff(col, p=1.0):
    open('./csv/'+str(int(p*100))+'%diff_'+col+'.csv', 'w').close()
    cluster = [ 
        MiniBatchKMeans(n_clusters=2)
        ,KMeans(n_clusters=2)
        ,SpectralClustering(n_clusters=2)
        ,Birch(n_clusters=2)
        ,AgglomerativeClustering(n_clusters=2)
    ]

    f = open('./csv/'+str(int(p*100))+'%diff_'+col+'.csv', 'a')
    f.write('video,clustering,acc\n')
    for img in lb.uvideo:
        x = lb.get_x(img, col)
        d = lb.get_diff_indata(x.T)
        d = np.where(d >= p)[0]
        x = x[:, d].reshape(-1, len(d))

        for c in cluster:
            c.fit(x)
            #print(img, type(c).__name__, get_accuracy(c.labels_, lb.get_l(img)), sep=',')
            f.write(str(img)+','+type(c).__name__+','+str(get_accuracy(c.labels_, lb.get_l(img, l=2)))+'\n')
    return



""" def hyper_parameter_perimg(img, n_data, ):
    
    return """