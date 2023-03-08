#import sklearn as sk
import lib as lb
import ml
import sys
import pickle
import time
import tsfel
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering #neki
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, OPTICS, BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.decomposition import KernelPCA, FactorAnalysis, FastICA, IncrementalPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB ## ne vem
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, StackingClassifier #najbulš s pravimi feature
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, butter, lfilter, filtfilt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mco
import itertools as it
## https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8861725/

#170108 16091401 16093601 16093801 """16092701"""

## np.zeros = decomposition na podatkih, vse videje skupaj v clustering
## algo za misssing values sklearn (sklearn.inpute. ...) ## posamezen video? idk, na vsak roi posebej bi mogli met label
##^https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
## forest v (model_selection)cv hyper parameter

## features drops, time to drop, 
##^^^^^^

#1400-ttpb.max()+ttp
for img in lb.uvideo:
    b = np.array(lb.data.loc[img].query("finding == 'Benign'")['NIR_nfp_smth'].values.tolist())
    if b.size > 0:
        ttpb = np.array(lb.data.loc[img].query("finding == 'Benign'")['TTP_smth'].values.tolist())
        for bx, ttp in zip(b, ttpb):
            #plt.plot(np.flip(bx[:ttp]))
            plt.plot(bx[ttp:], color='b')
        #plt.title('Benign')
        #plt.show()

    c = np.array(lb.data.loc[img].query("finding == 'Cancer'")['NIR_nfp_smth'].values.tolist())
    if c.size > 0:
        ttpb = np.array(lb.data.loc[img].query("finding == 'Cancer'")['TTP_smth'].values.tolist())
        for bx, ttp in zip(c, ttpb):
            #plt.plot(np.flip(bx[:ttp]))
            plt.plot(bx[ttp:], color='r')
        #plt.title('Cancer')
        #plt.show()

    h = np.array(lb.data.loc[img].query("finding == 'Healthy'")['NIR_nfp_smth'].values.tolist())
    ttpb = np.array(lb.data.loc[img].query("finding == 'Healthy'")['TTP_smth'].values.tolist())
    for bx, ttp in zip(h, ttpb):
        #plt.plot(np.flip(bx[:ttp]))
        plt.plot(bx[ttp:], color='g')
    #plt.title('Healthy')
    plt.show()



##idk
""" for img in lb.uvideo: ##največje razlike med podatki jih smooth in morde drops?
    x = lb.get_x(img, 'NIR_nfp_smth')
    ttp = lb.get_x(img, 'TTP_smth')
    dx = lb.get_diff_indata(x)
    #dx = dx[np.argmin(dx[ttp.min():])+ttp.min():] # od ttp naprej

    x = x[:, dx > 0.75]
    
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(x)

    print(img, ml.get_accuracy(model.labels_, lb.get_l(img, l=2)))
    #exit() """
#exit()