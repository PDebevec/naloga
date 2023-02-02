import numpy as np
import pandas as pd
data = np.load(open('data.npy', 'rb'), allow_pickle=True)

""" def to_array(dataframe):
    arr = []
    for element in dataframe:
        #arr.append(np.fromstring(element[1:-1], dtype=float, sep=','))
        #arr.append(list(float(x) for x in element[1:-1].split(',')))
        for x in element[1:-1].split(',')[:1400]:
            arr.append(float(x))
    return np.array(arr).reshape(1214, -1) """

def to_array(strs):
    for i,e in enumerate(strs):
        strs[i] = np.array([float(x) for x in e[1:-1].split(',')[:1400]])
    return strs

def get_image(arr):
    unique = []
    for e in np.unique(arr[:, 0, 0]):
        unique.append(np.where(arr[:, 0, 0] == e))
    images = []
    for i in unique:
        images.append(arr[i[0]])
    return images

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

def get_label(labels):
    return np.unique(labels)

def get_diff(data_arr):
    arr = []
    for d in data_arr:
        arr.append((d[0]-d[1])*-1)
        for i in range(1, len(d)-1):
            arr.append( ((d[i-1]-d[i]) + (d[i]-d[i+1])) * -0.5 )
        arr.append((d[len(d)-2]-d[len(d)-1])*-1)
    return np.array(arr).reshape(len(data_arr), 1400)

""" def get_diff_from(data_arr, fun):
    arr = []
    for d in data_arr:
        value = fun(d)
        arr.append((d[0]-d[1])*-1)
        for i in range(1, len(d)-1):
            arr.append( ((d[i-1]-value) + (value-d[i+1])) * -0.5 )
        arr.append((d[len(d)-2]-d[len(d)-1])*-1)
    return np.array(arr).reshape(1214, 1400) """

_separated_data = get_image(data)
_labels = get_label(data[:, 2, 0])
_separated_images = separate_image(_separated_data, _labels)
_time = np.arange(0, 1400*0.2002, 0.2002)