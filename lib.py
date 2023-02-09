import numpy as np
import pandas as pd
import pickle


class __lib():
    def __init__(self):
        self.fdata = pickle.load(open('fdata.pickle', 'rb'))
        self.fdata['NIR_minmax'] = self.get_normalized(self.fdata['NIR'])
        """ self.data = np.load(open('data.npy', 'rb'), allow_pickle=True)
        self._by_image = self.get_image()
        self._labels = self.get_label()
        self._image_by_label = self.labeled_image()
        self._time = np.arange(0, 1400*0.2002, 0.2002) """

    fdata = None
    data = None
    data2d = None
    _by_image = None
    _labels = None
    _image_by_label = None
    _time = None

    def get_image(self):
        img = []
        for e in np.unique(self.data[:, 0]):
            img.append(self.data[np.where(self.data[:, 0] == e)[0]])
        return img

    def get_label(self):
        return np.unique(self.data[:, 2])

    def labeled_image(self):
        labeled_img = []
        for img in self._by_image:
            labeled_img.append([])
            for label in self._labels:
                labeled_img[-1].append(img[np.where(img[:, 2] == label)])
        return labeled_img

    @staticmethod
    def get_diff(data_arr):
        arr = []
        for d in data_arr:
            arr.append((d[0]-d[1])*-1)
            for i in range(1, len(d)-1):
                arr.append( ((d[i-1]-d[i]) + (d[i]-d[i+1])) * -0.5 )
            arr.append((d[-2]-d[-1])*-1)
        return np.array(arr).reshape(len(data_arr), 1400)
    
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
    def time_of_curve():
        return 0
lib = __lib()

class __ml():

    @staticmethod
    def separate_labels(labels_, y_train):
        arr = []
        for label in np.unique(y_train):
            arr.append(list(labels_[np.where(y_train == label)]))
        return arr

    @staticmethod
    def find_biggest_batch(separate_labels, num):
        arr = []
        for label in separate_labels:
            arr.append( [[label.count(x), x] for x in np.unique(label)] )
            arr[-1] = sorted(arr[-1], key=lambda x: x[0])
            #arr[-1] = [ x[1] for x in arr[-1][-25:] ]
            arr[-1] = arr[-1][-num:]
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

    @staticmethod
    def get_xy_data(nirs):
        where = np.where(nirs[:, 2] == 'Healthy')
        xh = nirs[where, 4][0]
        xh = np.concatenate(xh).reshape(len(xh), 1400)
        yh = nirs[where, 2][0]
        where = np.where( nirs[:, 2] == 'Cancer')
        xc = nirs[where, 4][0]
        xc = np.concatenate(xc).reshape(len(xc), 1400)
        yc = nirs[where, 2][0]
        where = np.where(nirs[:, 2] == 'Benign')
        xb = nirs[where, 4][0]
        xb = np.concatenate(xb).reshape(len(xb), 1400)
        yb = nirs[where, 2][0]
        return np.concatenate((xh[:400], xc[:400])).ravel().reshape((-1, 1400)),\
        np.concatenate((xh[400:], xc[400:])).ravel().reshape((-1, 1400)),\
        np.concatenate((yh[:400], yc[:400])),\
        np.concatenate((yh[400:], yc[400:]))
        """ return np.concatenate((xh[:250], xc[:250], xb[:250])).ravel().reshape((-1, 1400)),\
        np.concatenate((xh[250:], xc[250:], xb[250:])).ravel().reshape((-1, 1400)),\
        np.concatenate((yh[:250], yc[:250], yb[:250])),\
        np.concatenate((yh[250:], yc[250:], yb[250:])) """
ml = __ml()