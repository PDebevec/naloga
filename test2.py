#import sklearn as sk
import ml
import sys
import time
import tsfel
import pickle
import lib as lb
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.colors as mco
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB ## 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, butter, lfilter, filtfilt
from scipy.optimize import curve_fit
###
##csv
#drops mean
#Ndata
#nir nfp butter
#csv od nevem.csv

""" for img in lb.uvideo:
    c = ''
    plt.title(img)
    for l,x in zip(lb.get_l(img), lb.get_x(img, 'NIR_nfp_butter')):
        match l:
            case 'Healthy': c = 'green'
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        plt.plot(x, color=c)
    plt.show() """


""" arr = []
for img in lb.uvideo:
    #plt.title(img)
    X = lb.get_x(img, 'NIR_nfp_butter').T
    ha = []
    i = 0
    a = []
    for x in X:
        x = x.reshape(-1, 1)
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(x)
        acc = ml.get_accuracy(model.labels_, lb.get_l(img, l=2))
        a.append(acc)
        if acc > 0.85:
            ha.append(i)
        i+=1
    arr.append(ha)
    plt.title(str(img) + '\nmin:' + str(int(np.min(a)*1000)/10) + '% max:' + str(int(np.max(a)*1000)/10) + '% mean:' + str(int(np.mean(a)*1000)/10) + '%')
    plt.plot(a)
    plt.plot(lb.get_diff_indata_minmax(X.T))
    plt.show() """



#exit()