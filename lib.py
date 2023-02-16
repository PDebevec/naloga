import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d

class __lib():
    def __init__(self):
        self.data = pickle.load(open('fdata.pickle', 'rb'))
        self.data['NIR_255'] = self.data['NIR']/255
        self.data['NIR_minmax'] = self.get_minmax(self.data['NIR'])
        self.data['NIR_minmax_img'] = self.get_minmax_byimg(self.data['NIR'])
        self.data['NIR_diff'] = self.get_diff(self.data['NIR'])

        self.data['NIR_255_smth'] = self.get_gaussian(self.data['NIR_255'].values, 15)
        self.data['NIR_minmax_smth'] = self.get_gaussian(self.data['NIR_minmax'].values, 15)
        self.data['NIR_minmax_img_smth'] = self.get_gaussian(self.data['NIR_minmax_img'].values, 15)
        self.data['NIR_diff_smth'] = self.get_gaussian(self.data['NIR_diff'].values, 15)

        """ self.data['NIR_smth_255'] = [ [ y/255 for y in x ] for x in np.array(self.get_gaussian(self.data['NIR'], 15)) ]
        self.data['NIR_smth_minmax'] = self.get_minmax(self.get_gaussian(self.data['NIR_255'], 15))
        self.data['NIR_smth_minmax_img'] = self.get_minmax_byimg(self.data['NIR_255_smth'])
        self.data['NIR_smth_diff'] = self.get_diff(self.get_gaussian(self.data['NIR_255'], 15)) """
    data = None

    @staticmethod
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
    
    @staticmethod
    def get_minmax(data):
        arr = []
        for x in data:
            mx = np.max(x)
            mn = np.min(x)
            arr.append( np.array((x - mn) / (mx - mn)) )
        return arr

    @staticmethod
    def get_minmax_byimg(data):
        arr = []
        for img in pd.unique(data.index.get_level_values(0)):
            mx = np.max( [max(x) for x in data.loc[img].values] )
            mn = np.min( [min(x) for x in data.loc[img].values] )
            for x in data.loc[img]:
                arr.append( np.array((x - mn) / (mx - mn)) )
        return arr

    @staticmethod
    def get_lables(data_lables):
        arr = []
        for img in pd.unique(data_lables.index.get_level_values(0)):
            ulabels = pd.unique(data_lables.loc[img].index.get_level_values(0))
            for label in data_lables.loc[img].index.get_level_values(0):
                arr.append(np.where(ulabels == label))
        return np.concatenate(arr)
    
    @staticmethod
    def get_img_label(data):
        arr = []
        for img in pd.unique(data.index.get_level_values(0)):
            ulabels = pd.unique(data.loc[img].index.get_level_values(0))
            if 'Benign' in ulabels:
                arr.append(0)
            else:
                arr.append(1)
        return arr
    
    @staticmethod
    def seperate_bylabel(lable1, label2, values):
        l1 = np.logical_or(lable1 == 0, label2 == 0)
        l2 = np.logical_or(lable1 == 1, label2 == 1)
        print(l1.shape, l2.shape)
        arr = np.concatenate(values).reshape(-1, 1400)
        return list(arr[:, l1]), list(arr[:, l2])
    
    @staticmethod
    def get_gaussian(data, sigma):
        arr = []
        for x in data:
            arr.append(gaussian_filter1d(x, sigma))
        return arr

    @staticmethod
    def get_avg(data):
        avg = []
        for y in range(len(data[0])):
            avg.append( np.average( [ data[x][y] for x in range(len(data)) ] ) )
        return avg
lib = __lib()

class __ml():

    @staticmethod
    def run_clustering(x, img, clusteringfun, column):
        x = x.loc[img]

        model = clusteringfun(n_clusters=2)
        model.fit(np.concatenate(x[column].values).reshape(-1, len(x[column][0])))

        #print(type(model))
        #return [type(model).__name__, ml.get_accuracy(model.labels_, x.index.get_level_values(0))]
        return ml.get_accuracy(model.labels_, x.index.get_level_values(0))
        """ sl = ml.separate_labels(model.labels_, x.index.get_level_values(0))
        bil = ml.find_batch_inlabel(sl, model.labels_, np.unique(x.index.get_level_values(0))) """
        """ corect = 0
        for i in range(len(x.index.get_level_values(0))):
            #print(bil[i], data.index.get_level_values(0)[i])
            if bil[i] == x.index.get_level_values(0)[i]:
                corect+=1
        print(corect, len(bil), corect/len(bil)) """

    @staticmethod
    def get_accuracy(labels_, y):
        arr = []
        model = LabelBinarizer()
        res = np.concatenate(model.fit_transform(y))
        acc = accuracy_score(res, labels_)
        return max([ acc, (acc-1)*-1 ])

    @staticmethod
    def separate_labels(labels_, y_train):
        arr = []
        for label in np.unique(y_train):
            arr.append(list(labels_[np.where(y_train == label)]))
        separate_labels = arr
        for x in np.unique(np.concatenate(separate_labels)):
            arr[np.argmax([ y.count(x) for y in separate_labels])].append(x)
        return arr

    @staticmethod
    def find_batch_inlabel(separate_labels, labels_, ulabels):
        arr = []
        for x in labels_:
            arr.append(ulabels[np.argmax([ y.count(x) for y in separate_labels ])])
        return arr

    @staticmethod
    def find_batch(labels_, labeled, labels):
        arr = []
        for batch in labels_:
            arr.append(labels[np.argmax([x.count(batch) for x in labeled])])
        return arr
    
    @staticmethod
    def to_array(strs):
        for i,e in enumerate(strs):
            strs[i] = np.array([float(x) for x in e[1:-1].split(',')])[:1400]
        return strs
ml = __ml()