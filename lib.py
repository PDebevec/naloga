import numpy as np
import pandas as pd
data = np.load(open('data.npy', 'rb'), allow_pickle=True)

def get_image(arr):
    unique = []
    for e in np.unique(arr[:, 0, 0]):
        unique.append(np.where(arr[:, 0, 0] == e))
    images = []
    for i in unique:
        images.append(arr[i[0]])
    return images

def get_label(labels):
    return np.unique(labels)

def separate_image(arr_img, labels):
    image_by_label = []
    for img in arr_img:
        temp = []
        for label in labels:
            temp2 = []
            for x in np.where(img[:, 2, 0] == label):
                for y in x:
                    temp2.append(img[y])
            temp.append(temp2)
        image_by_label.append(temp)
    return image_by_label

_separated_data = get_image(data)
_labels = get_label(data[:, 2, 0])
_separated_images = separate_image(_separated_data, _labels)
_time = np.arange(0, 1400*0.2002, 0.2002)

def separate_labels(labels_, y_train):
    arr = []
    for label in np.unique(y_train):
        arr.append(list(labels_[np.where(y_train == label)]))
    return arr

def find_batch(labels_, labeled, labels=_labels):
    arr = []
    for batch in labels_:
        arr.append(labels[np.argmax([
            labeled[0].count(batch),
            labeled[1].count(batch),
            #labeled[2].count(batch)
        ])])
    return arr

def to_array(strs):
    for i,e in enumerate(strs):
        strs[i] = np.array([float(x) for x in e[1:-1].split(',')[:1400]])
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
