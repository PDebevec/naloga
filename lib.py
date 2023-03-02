import numpy as np
import pandas as pd
import pickle
import tsfel
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, TruncatedSVD
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

#data1 = pickle.load(open('data1.pickle', 'rb'))
#data2 = pickle.load(open('data2.pickle', 'rb'))
data = pickle.load(open('data.pickle', 'rb'))
videos = pickle.load(open('videolabel.pickle', 'rb'))
uvideo = pd.unique(data.index.get_level_values(0))
ulabel = videos['label'].values

def to_array(strs, num=1400):
    for i,e in enumerate(strs):
        strs[i] = np.array([float(x) for x in e[1:-1].split(',')])[:num]
    return strs

def get_x(img, col):
    return np.array(data.loc[img][col].values.tolist())

def get_allx(col):
    return np.array(data[col].values.tolist())

def get_l(img, l=0):
    return data.loc[img].index.get_level_values(l)

def get_diff_indata(X):
    arr = []
    X = np.array([
        y/y.mean() for y in X+1
    ])
    X -= X.min()
    X /= X.max()
    X = (X.max(axis=1) - X.min(axis=1))
    return X/X.max()

def get_num_of_data(X, num):
    arr = []
    for x in X:
        arr.append(x[::int(len(x)/num)])
    return arr

""" def get_diff(data_arr):
    arr = []
    for d in data_arr:
        temp = []
        temp.append(-(d[0]-d[1]))
        for i in range(1, len(d)-1):
            temp.append( ((d[i-1]-d[i]) + (d[i]-d[i+1])) * -0.5 )
        temp.append(-(d[-2]-d[-1]))
        arr.append(temp)
    return arr """

def get_minmax(X):
    arr = []
    for x in X:
        mx = np.max(x)
        mn = np.min(x)
        arr.append( np.array((x - mn) / (mx - mn)) )
    return arr

def get_nfp(X):
    arr = []
    for x in X:
        n = x[find_peaks(x, distance=250, height=0.5)[0][0]]
        arr.append(x/n)
    return arr

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
        res = model.fit_transform(X.loc[img].index.get_level_values(0))+1
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

def get_gaussian_diff(X, sigma):
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

def reject_outliers(X, m=2):
    return X[abs(X - np.mean(X)) < m * np.std(X)]


""" data['NIR_255'] = data['NIR']/255
data['NIR_minmax'] = get_minmax(data['NIR'])
data['NIR_minmax_img'] = get_minmax_byimg(data['NIR'])
data['NIR_diff'] = get_diff(data['NIR'])

data['NIR_255_smth'] = get_gaussian(data['NIR_255'].values, 15)
data['NIR_minmax_smth'] = get_gaussian(data['NIR_minmax'].values, 15)
data['NIR_minmax_img_smth'] = get_gaussian(data['NIR_minmax_img'].values, 15)
data['NIR_diff_smth'] = get_gaussian(data['NIR_diff'].values, 15) """