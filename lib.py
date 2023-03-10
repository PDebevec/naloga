import ml
import sys
import pickle
import time
import tsfel
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB ## blo v poslanem (link spodi)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, StackingClassifier #najbulš s pravimi feature
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, butter, lfilter, filtfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import itertools as it


#data1 = pickle.load(open('data1.pickle', 'rb'))
#data2 = pickle.load(open('data2.pickle', 'rb'))
data = pickle.load(open('data.pickle', 'rb'))
videos = pickle.load(open('videolabel.pickle', 'rb'))
uvideo = pd.unique(data.index.get_level_values(0))
ulabel = videos['label'].values
tsd = pickle.load(open('tsd.pickle', 'rb'))

def to_array(strs, num=1490):
    for i,e in enumerate(strs):
        strs[i] = np.array([float(x) for x in e[1:-1].split(',')])[:num]
    return strs

def get_x(img, col):
    return np.array(data.loc[img][col].values.tolist())

def get_dfx(img, col):
    return data.loc[img][col]

def get_allx(col):
    return np.array(data[col].values.tolist())

def get_l(img, l=0):
    return np.array(data.loc[img].index.get_level_values(l))

def get_alll(l=0):
    return np.array(data.index.get_level_values(l))

def reject_outliers(X, m=2):
    return X[abs(X - np.mean(X)) < m * np.std(X)]

def get_diff_indata(X):
    X = np.array([
        y/y.mean() for y in X.T+1
    ])
    X -= X.min()
    X /= X.mean()
    X = (X.mean(axis=1) - X.min(axis=1))
    return X#/X.max()

def get_diff_indata_minmax(X):
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    X = mx-mn
    return X

def get_drop_mean(X):
    arr = []
    arr1 = []
    for r in X:
        i = np.where(r == 1)[0][0]
        r = r[i:]
        #print(r[::int(len(r)/5)-1])
        #print(np.arange(0, len(r), int(len(r)/6)-1)[1:-1]+i)
        i = np.arange(0, len(r), int(len(r)/8)-1)[1:-1]
        arr.append(r[i])
        arr1.append([ np.mean(r[ri-j*10:ri+j*10]) for j,ri in enumerate(i, 1) ])
    return arr, arr1

def get_data_fromdiff(col, per=0.5):
    for img in uvideo:
        x = get_dfx(img, col)
        x_train = get_x(img, col)

        d = get_diff_indata(x_train)
        temp = 0
        #temp = np.argmin(d[50:])+50
        #d = d[temp:]
        #d -= d[0]
        x_train = x_train[:, np.where(d >= d.max()*per)[0]+temp ]

def get_num_of_data(X, num):
    arr = []
    for x in X:
        arr.append(x[::int(len(x)/num)])
    return arr

def get_diff_peak(X, TTP):
    arr = []
    for x,ttp in zip(X, TTP):
        temp = []
        p = x[ttp]
        temp.append(x[0])
        for i in range(1, len(x)-1):
            temp.append( ((x[i-1]-p) + (p-x[i+1])) * -0.5 )
        temp.append(p-x[-1])
        arr.append(temp)
    return arr

def get_minmax(X):
    arr = []
    for x in X:
        mx = np.max(x)
        mn = np.min(x)
        arr.append( np.array((x - mn) / (mx - mn)) )
    return arr

def get_nfp(X):
    arr = []
    arr2 = []
    for x in X:
        #print(find_peaks(x, distance=200, height=np.max(x)*0.5))
        #p = find_peaks(x, distance=150, height=np.max(x)*0.6)[0][0]
        p = find_peaks(x)[0][0]
        mx = x[p]
        mn = x.min()
        #arr.append(x/n)
        arr.append( np.array((x - mn) / (mx - mn)) )
        arr2.append(p)
    return arr, arr2

""" def get_tt_mm(X, TTP):
    arr = []
    arr1 = []
    for x,ttp in zip(X, TTP):
        arr.append( np.where(x[ttp:] == x[ttp:].min())[0]+ttp )
        arr1.append( np.where(x[ttp:] == x[ttp:].max())[0]+ttp )
    return arr, arr1 """

""" def get_shift_nfp(X):
    arr = []
    for img in pd.unique(X.index.get_level_values(0)):
        temp = None
        ffp = []
        for x in X.loc[img].values:
            fp = find_peaks(x[:300], distance=25, height=np.max(x[:300])*0.5)[0][0]
            n = x[fp]
            ffp.append(fp)
            temp = x/n
            temp = np.roll(temp, ffp[0]-fp)
            arr.append(temp)
    return arr """

""" def get_minmax_byimg(data):
    arr = []
    for img in pd.unique(data.index.get_level_values(0)):
        mx = np.max( [max(x) for x in data.loc[img].values] )
        mn = np.min( [min(x) for x in data.loc[img].values] )
        for x in data.loc[img]:
            arr.append( np.array((x - mn) / (mx - mn)) )
    return arr """

def get_binary(X):
    arr = []
    for img in pd.unique(X.index.get_level_values(0)):
        model = LabelBinarizer()
        res = model.fit_transform(X.loc[img].index.get_level_values(0))
        arr.append(res.reshape(-1).tolist())
    return sum(arr, [])

def get_lables(data_lables):
    arr = []
    for img in pd.unique(data_lables.index.get_level_values(0)):
        ulabels = pd.unique(data_lables.loc[img].index.get_level_values(0))
        for label in data_lables.loc[img].index.get_level_values(0):
            arr.append(np.where(ulabels == label))
    return np.concatenate(arr)


def get_img_label(X):
    arr = []
    for img in pd.unique(X.index.get_level_values(0)):
        ulabels = pd.unique(X.loc[img].index.get_level_values(0))
        if 'Benign' in ulabels:
            arr.append(0)
        else:
            arr.append(1)
    return arr

def get_gaussian(X, sigma):
    arr = []
    for x in X:
        arr.append(gaussian_filter1d(x, sigma))
        #arr[-1] = [ x/arr[-1].max() for x in arr[-1]]
        arr[-1] /= arr[-1].max()
    return arr

def get_savgol(X):
    arr = []
    for x in X:
        arr.append(savgol_filter(x, int(len(X)/14), 5))
        arr[-1] /= arr[-1].max()
    return arr

def get_butter(X):
    arr = []
    for x in X:
        b, a = butter(2, 0.0075)
        arr.append(lfilter(b, a, x))
    return arr

def get_gaussian_diff(X, sigma=1):
    arr = []
    for x in X:
        arr.append(gaussian_filter1d(x, sigma=sigma, order=1, mode='nearest'))
        #arr[-1] = [ x/arr[-1].max() for x in arr[-1]]
        arr[-1] /= arr[-1].max()
    return arr

def get_avg(X):
    avg = []
    for y in range(len(X[0])):
        avg.append( np.average( [ X[x][y] for x in range(len(X)) ] ) )
    return avg

def get_tsfd(X):
    model = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain(),
        X.to_numpy(),
        #fs=2, #ne vpliva na podatke ali hitrost default=None
        verbose=1)
    return model.values.tolist()

def get_divisor(num):
    arr = []
    for i in range(1, int(num/2+1)):
        if num % i == 0:
            arr.append(i)
    return arr



""" data['NIR_255'] = data['NIR']/255
data['NIR_minmax'] = get_minmax(data['NIR'])
data['NIR_minmax_img'] = get_minmax_byimg(data['NIR'])
data['NIR_diff'] = get_diff(data['NIR'])

data['NIR_255_smth'] = get_gaussian(data['NIR_255'].values, 15)
data['NIR_minmax_smth'] = get_gaussian(data['NIR_minmax'].values, 15)
data['NIR_minmax_img_smth'] = get_gaussian(data['NIR_minmax_img'].values, 15)
data['NIR_diff_smth'] = get_gaussian(data['NIR_diff'].values, 15) """