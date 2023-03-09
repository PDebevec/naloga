import numpy as np
import lib as lb
import ml
import matplotlib.pyplot as plt
import scipy.signal as ss
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering, Birch


#izris vsake slike posebej in shrani
""" for img in pd.unique(lb.uvideo):
    print(img)
    #fig, axis = plt.subplots(1,2)
    for l,x in zip(lb.get_l(img), lb.get_x(img, 'NIR_nfp_smth')):
        c = ''
        match l:
            case 'Healthy': c = 'green'
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        plt.plot(x, color=c)
    #plt.savefig('graphs/nfp/'+str(img)+'_nfp.png')
    #plt.savefig('graphs/'+str(img)+'.png')
    plt.show()
    #exit() """
#simple
""" for img in pd.unique(lb.uvideo):
    c = ''
    plt.title(img)
    for l,x in zip(lb.get_l(img), lb.get_x(img, 'NIR_nfp_smth')):
        match l:
            case 'Healthy': c = 'green'
            case 'Benign': c = 'blue'
            case 'Cancer': c = 'red'
        plt.plot(x, color=c)
    plt.show() """

# diff in data za vsak video """ od ttp naprej < morem še dodat """
""" c = pd.unique(lb.data.query("finding == 'Cancer'").index.get_level_values(0))
for img in c:
    x = lb.get_x(img, 'NIR_nfp_butter')
    x = lb.get_diff_indata(x)
    plt.plot(x, label=img)
    plt.legend(loc='lower right')
    #plt.plot(x.T)
plt.show() """

#razlika med podatki na minmax benign/cancer
""" for img in lb.uvideo:
    x = lb.get_diff_indata_minmax(lb.get_x(img, 'NIR_nfp_butter'))
    if lb.videos.loc[img]['label'] == 'Benign':
        plt.plot(x, color='b')
    else:
        plt.plot(x, color='r')
plt.show() """

#posebej benign calncer in healthy
""" b = np.array(lb.data.query("finding == 'Benign'")['NIR_nfp_smth'].values.tolist())
ttpb = np.array(lb.data.query("finding == 'Benign'")['TTP_smth'].values.tolist())
for bx, ttp in zip(b, ttpb):
    #plt.plot(np.flip(bx[:ttp]))
    plt.plot(bx[ttp:1400-ttpb.max()+ttp])
plt.title('Benign')
plt.show()

c = np.array(lb.data.query("finding == 'Cancer'")['NIR_nfp_smth'].values.tolist())
ttpb = np.array(lb.data.query("finding == 'Cancer'")['TTP_smth'].values.tolist())
for bx, ttp in zip(c, ttpb):
    #plt.plot(np.flip(bx[:ttp]))
    plt.plot(bx[ttp:1400-ttpb.max()+ttp])
plt.title('Cancer')
plt.show()

h = np.array(lb.data.query("finding == 'Healthy'")['NIR_nfp_smth'].values.tolist())
ttpb = np.array(lb.data.query("finding == 'Healthy'")['TTP_smth'].values.tolist())
for bx, ttp in zip(h, ttpb):
    #plt.plot(np.flip(bx[:ttp]))
    plt.plot(bx[ttp:1400-ttpb.max()+ttp])
plt.title('Healthy')
plt.show() """
#imgs = [170108?, 16090101, 16092101?, 16092201, 16093801?] #problemi s ttp
#od ttp naprej oz obratno
""" for img in lb.uvideo:
    plt.title(img)
    col = 'NIR_nfp_butter'
    tcol = 'TTP_butter'
    b = np.array(lb.data.loc[img].query("finding == 'Benign'")[col].values.tolist())
    if b.size > 0:
        ttpb = np.array(lb.data.loc[img].query("finding == 'Benign'")[tcol].values.tolist())
        for bx, ttp in zip(b, ttpb):
            plt.plot(np.flip(bx[:ttp]), color='b')
            #1400-ttpb.max()+ttp # za odrezat podatke
            #plt.plot(bx[ttp:], color='b')
        #plt.title('Benign')
        #plt.show()

    c = np.array(lb.data.loc[img].query("finding == 'Cancer'")[col].values.tolist())
    if c.size > 0:
        ttpb = np.array(lb.data.loc[img].query("finding == 'Cancer'")[tcol].values.tolist())
        for bx, ttp in zip(c, ttpb):
            plt.plot(np.flip(bx[:ttp]), color='r')
            #plt.plot(bx[ttp:], color='r')
        #plt.title('Cancer')
        #plt.show()

    h = np.array(lb.data.loc[img].query("finding == 'Healthy'")[col].values.tolist())
    ttpb = np.array(lb.data.loc[img].query("finding == 'Healthy'")[tcol].values.tolist())
    for bx, ttp in zip(h, ttpb):
        plt.plot(np.flip(bx[:ttp]), color='g')
        #plt.plot(bx[ttp:], color='g')
    #plt.title('Healthy')
    plt.show() """

# podatki iz katerih dobiš >90% acc
""" for img in lb.uvideo:
    plt.title(img)
    X = lb.get_x(img, 'NIR_nfp_smth').T
    ha = []
    i = 0
    for x in X:
        x = x.reshape(-1, 1)
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(x)
        acc = ml.get_accuracy(model.labels_, lb.get_l(img, l=2))
        if acc > 0.9:
            #print(acc)
            ha.append(i)
        i+=1
    #print(ha)
    plt.plot(X[ha, :])
    plt.show() """
# še za tsd
""" for img in lb.uvideo:
    plt.title(img)
    X = lb.tsd.loc[img].values.T
    ha = []
    i = 0
    for x in X:
        x = x.reshape(-1, 1)
        if np.isnan(x).any():
            i+=1
            continue
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(x)
        acc = ml.get_accuracy(model.labels_, lb.get_l(img, l=2))
        if acc > 0.95:
            #print(acc)
            ha.append(i)
        i+=1
    #print(ha)
    plt.plot(X[ha, :])
    plt.show() """
#prikaz acc in diff v podatkih
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
    plt.plot(lb.get_diff_indata(X.T))
    plt.show() """


#3d graf slik
""" for img in pd.unique(lb.data.index.get_level_values(0)):
    z = np.concatenate(lb.data.loc[img]['NIR_255'].values).reshape(-1, 1400)
    x = np.array( [ np.full((1400), i) for i in range(len(z)) ] )
    y = np.array( [ np.arange(1400) for i in range(len(z)) ] )
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(x, y, z, color='green')
    plt.show() """

#ttp
""" for img in range(1000,1002):
    plt.plot(lb.data[img, 4][:300])
    am = np.argmax(lb.data[img, 4][:300])
    plt.plot(am, lb.data[img, 4][am], 'x')
    ttp = int(am *0.05)
    plt.plot(ttp, lb.data[img, 4][ttp], 'x')
    #plt.plot(int(lb.data[img, 3]/0.2002), lb.data[img, 4][int(lb.data[img, 3]/0.2002)], 'x') """

#ttp
""" #graf = 100
graf = 1200
ttp = ss.find_peaks(lb.data[graf, 4][:300], distance=200)

#print(ttp[0] * 0.2002, data[graf, 3, 0])
plt.plot(lb.data[graf, 4])
for e in ttp[0]:
    plt.plot(e, lb.data[graf, 4][e], 'x')

plt.axvline(int(lb.data[graf, 3]/0.2002))
plt.axvline(ttp[0])
plt.axvline(ttp[0] - int(lb.data[graf, 3]/0.2002)) """

#navečja razlika v nir in peak
""" graf = 100
dif = lb.get_diff(lb.data[:, 4][:500])[graf]
plt.plot(dif)
plt.plot(lb.data[:, 4][graf] / 35)
p = ss.find_peaks(dif, height=0, distance=200)[0]
d = ss.find_peaks(-dif, height=-1, distance=1)[0]
plt.plot(p, dif[p], 'x', color='black')
plt.plot(d, dif[d], 'x', color='red') """

#za posamezne label
""" figure, axis = plt.subplots(5)
img = 5+10

index = 0
for x in lb._image_by_label[10:img]:
    print(len(x[0]), len(x[1]), len(x[2]))
    for e in x[1]:
        axis[index].plot(e[4], color='red')
    for e in x[0]:
        axis[index].plot(e[4], color='blue')
    for e in x[2]:
        axis[index].plot(e[4], color='green')
    index+=1 """

# ^ 23 slika
""" img = 23
for e in lb._image_by_label[img][0]:
    plt.plot(e[4], color='blue')
for e in lb._image_by_label[img][1]:
    plt.plot(e[4], color='red')
for e in lb._image_by_label[img][2]:
    plt.plot(e[4], color='green') """

#za posamezne label 2
""" figure, axis = plt.subplots(2,2)
img = 24

for e in lb._image_by_label[img][2]:
    axis[0, 0].plot(e[4], color='green')
    axis[1, 0].plot(e[4], color='green')
for e in lb._image_by_label[img][1]:
    axis[0, 1].plot(e[4], color='red')
for e in lb._image_by_label[img][0]:
    axis[1, 1].plot(e[4], color='blue') """

#za posamezen img
""" fiture, axis = plt.subplots(3,2)
for i,e in enumerate(lb._by_image[:3]):
    for k in e[:, 4]:
        axis[i, 0].plot(k)
    for k in e[:, 6]:
        axis[i, 1].plot(k) """

#smooth
""" graf = 120
d = ss.savgol_filter(lb.data[graf, 4][:300], window_length=20, polyorder=2)

for x in np.where(lb.data[:, 2] == 'Cancer')[0][::10]:
    plt.plot(lb.data[x, 4][:300], color='red')
for x in np.where(lb.data[:, 2] == 'Healthy')[0][::10]:
    plt.plot(lb.data[x, 4][:300], color='green')
for x in np.where(lb.data[:, 2] == 'Benign')[0][::10]:
    plt.plot(lb.data[x, 4][:300], color='blue') """

#plt.show()
