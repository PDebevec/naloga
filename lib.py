import numpy as np
import pandas as pd


def to_array(dataframe):
    arr = []
    for element in dataframe:
        #arr.append(np.fromstring(element[1:-1], dtype=float, sep=','))
        arr.append(list(float(x) for x in element[1:-1].split(',')))
    return arr

def get_image(data):
    unique = []
    for e in np.unique(data[:, 0]):
        unique.append(np.where(data[:, 0] == e))
    images = []
    for i in unique:
        images.append(data[i[0]])
    return images

def seperate_image(data, column, labels):
    image_per_label = []
    for i in data:
        for j in i[:, column]:
            for l in labels:
                image_per_label.append()
    return 0

def get_norm(data):
    for i, e in enumerate(data):
        data[i] = (np.array(e)-min(data.min()))/(max(data.max())-min(data.min()))
    return data

def get_label(data):
    return np.unique(data)