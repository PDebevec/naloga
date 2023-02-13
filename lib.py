import numpy as np
import pandas as pd
import pickle


class __lib():
    def __init__(self):
        self.data = pickle.load(open('fdata.pickle', 'rb'))
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
    def get_normalized(data):
        arr = []
        for x in data:
            mx = np.max(x)
            mn = np.min(x)
            arr.append( np.array((x - mn) / (mx - mn)) )
        return arr

    @staticmethod
    def get_normalized_byimg(data):
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
lib = __lib()

class __ml():

    @staticmethod
    def run_clustering(x, img, clusteringfun, column, time):
        x = x.loc[img]

        model = clusteringfun(n_clusters=2)
        res = model.fit_predict(np.concatenate(x[column].values).reshape(-1, 1400)[:, :time])
        
        print(type(model))
        #print(res.labels_)
        sl = ml.separate_labels(model.labels_, x.index.get_level_values(0))
        bil = ml.find_batch_inlabel(sl, model.labels_, np.unique(x.index.get_level_values(0)))

        corect = 0
        for i in range(len(x.index.get_level_values(0))):
            #print(bil[i], data.index.get_level_values(0)[i])
            if bil[i] == x.index.get_level_values(0)[i]:
                corect+=1
        print(corect, len(bil), corect/len(bil))
        return

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