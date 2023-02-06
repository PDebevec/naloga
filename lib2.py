import numpy as np
import pandas as pd
data = np.load(open('data.npy', 'rb'), allow_pickle=True)

def get_image(arr=data):
    unique = []
    for e in np.unique(arr[:, 0, 0]):
        unique.append(arr[np.where(arr[:, 0, 0] == e)[0]])
    return unique

def get_label(labels):
    return np.unique(labels)

_by_image = get_image(data)
_labels = get_label(data[:, 2, 0])

def labeled_image(arr_img=_by_image, labels=_labels):
    labeled_img = []
    for img in arr_img:
        labeled_img.append([])
        for label in labels:
            labeled_img[-1].append(img[np.where(img[:, 2, 0] == label)])
    return labeled_img

_image_by_label = labeled_image(_by_image, _labels)
_time = np.arange(0, 1400*0.2002, 0.2002)

def separate_labels(labels_, y_train):
    arr = []
    for label in np.unique(y_train):
        arr.append(list(labels_[np.where(y_train == label)]))
    return arr

def find_batch(labels_, labeled, labels=_labels):
    arr = []
    for batch in labels_:
        arr.append(labels[np.argmax([x.count(batch) for x in labeled])])
    return arr

def to_array(strs):
    for i,e in enumerate(strs):
        strs[i] = np.array([float(x) for x in e[1:-1].split(',')])[:1400]
    return strs

def get_diff(data_arr):
    arr = []
    for d in data_arr:
        arr.append((d[0]-d[1])*-1)
        for i in range(1, len(d)-1):
            arr.append( ((d[i-1]-d[i]) + (d[i]-d[i+1])) * -0.5 )
        arr.append((d[-2]-d[-1])*-1)
    return np.array(arr).reshape(len(data_arr), 1400)

def get_xy_data(nirs = data):
    where = np.where(nirs[:, 2, 0] == 'Healthy')
    xh = nirs[where, 4, 0][0]
    xh = np.concatenate(xh).reshape(len(xh), 1400)
    yh = nirs[where, 2, 0][0]
    where = np.where( nirs[:, 2, 0] == 'Cancer')
    xc = nirs[where, 4, 0][0]
    xc = np.concatenate(xc).reshape(len(xc), 1400)
    yc = nirs[where, 2, 0][0]
    where = np.where(nirs[:, 2, 0] == 'Benign')
    xb = nirs[where, 4, 0][0]
    xb = np.concatenate(xb).reshape(len(xb), 1400)
    yb = nirs[where, 2, 0][0]
    return np.concatenate((xh[:400], xc[:400])).ravel().reshape((-1, 1400)),\
    np.concatenate((xh[400:], xc[400:])).ravel().reshape((-1, 1400)),\
    np.concatenate((yh[:400], yc[:400])),\
    np.concatenate((yh[400:], yc[400:]))
    """ return np.concatenate((xh[:250], xc[:250], xb[:250])).ravel().reshape((-1, 1400)),\
    np.concatenate((xh[250:], xc[250:], xb[250:])).ravel().reshape((-1, 1400)),\
    np.concatenate((yh[:250], yc[:250], yb[:250])),\
    np.concatenate((yh[250:], yc[250:], yb[250:])) """
