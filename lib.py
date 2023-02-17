import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d

data = None

def get_diff(data_arr):
    arr = []
    for d in data_arr:
        temp = []
        temp.append(-(d[0]-d[1]))
        for i in range(1, len(d)-1):
            temp.append( ((d[i-1]-d[i]) + (d[i]-d[i+1])) * -0.5 )
        temp.append(-(d[-2]-d[-1]))
        arr.append(temp)
    return arr


def get_minmax(data):
    arr = []
    for x in data:
        mx = np.max(x)
        mn = np.min(x)
        arr.append( np.array((x - mn) / (mx - mn)) )
    return arr


def get_minmax_byimg(data):
    arr = []
    for img in pd.unique(data.index.get_level_values(0)):
        mx = np.max( [max(x) for x in data.loc[img].values] )
        mn = np.min( [min(x) for x in data.loc[img].values] )
        for x in data.loc[img]:
            arr.append( np.array((x - mn) / (mx - mn)) )
    return arr


def get_lables(data_lables):
    arr = []
    for img in pd.unique(data_lables.index.get_level_values(0)):
        ulabels = pd.unique(data_lables.loc[img].index.get_level_values(0))
        for label in data_lables.loc[img].index.get_level_values(0):
            arr.append(np.where(ulabels == label))
    return np.concatenate(arr)


def get_img_label(data):
    arr = []
    for img in pd.unique(data.index.get_level_values(0)):
        ulabels = pd.unique(data.loc[img].index.get_level_values(0))
        if 'Benign' in ulabels:
            arr.append(0)
        else:
            arr.append(1)
    return arr


def seperate_bylabel(lable1, label2, values):
    l1 = np.logical_or(lable1 == 0, label2 == 0)
    l2 = np.logical_or(lable1 == 1, label2 == 1)
    print(l1.shape, l2.shape)
    arr = np.concatenate(values).reshape(-1, 1400)
    return list(arr[:, l1]), list(arr[:, l2])


def get_gaussian(data, sigma):
    arr = []
    for x in data:
        arr.append(gaussian_filter1d(x, sigma))
    return arr


def get_avg(data):
    avg = []
    for y in range(len(data[0])):
        avg.append( np.average( [ data[x][y] for x in range(len(data)) ] ) )
    return avg

data = pickle.load(open('fdata.pickle', 'rb'))
data['NIR_255'] = data['NIR']/255
data['NIR_minmax'] = get_minmax(data['NIR'])
data['NIR_minmax_img'] = get_minmax_byimg(data['NIR'])
data['NIR_diff'] = get_diff(data['NIR'])

data['NIR_255_smth'] = get_gaussian(data['NIR_255'].values, 15)
data['NIR_minmax_smth'] = get_gaussian(data['NIR_minmax'].values, 15)
data['NIR_minmax_img_smth'] = get_gaussian(data['NIR_minmax_img'].values, 15)
data['NIR_diff_smth'] = get_gaussian(data['NIR_diff'].values, 15)