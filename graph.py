import numpy as np
from lib import lib as lb
import matplotlib.pyplot as plt
import scipy.signal as ss

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
figure, axis = plt.subplots(5)
img = 5+20

index = 0
for x in lb._image_by_label[20:img]:
    for e in x[2]:
        axis[index].plot(e[4], color='green')
    for e in x[1]:
        axis[index].plot(e[4], color='red')
    for e in x[0]:
        axis[index].plot(e[4], color='blue')
    index+=1

# ^ 23 slika
""" for e in lb._image_by_label[23][0]:
    plt.plot(e[4], color='blue')
for e in lb._image_by_label[23][1]:
    plt.plot(e[4], color='red')
for e in lb._image_by_label[23][2]:
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

plt.show()
