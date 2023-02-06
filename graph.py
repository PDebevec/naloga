import numpy as np
from lib import lib
lb = lib()
import matplotlib.pyplot as plt
import scipy.signal as ss

#ttp
""" for img in range(1):
    plt.plot(lb.data[img, 4][:300])
    am = np.argmax(lb.data[img, 4][:300])
    plt.plot(am, lb.data[img, 4][am], 'x')
    ttp = int(am *0.1)
    plt.plot(ttp, lb.data[img, 4][ttp], 'x')
    plt.plot(int(lb.data[img, 3]/0.2002), lb.data[img, 4][int(lb.data[img, 3]/0.2002)], 'x') """

#ttp
""" graf = 0
ttp = ss.find_peaks(lb.data[graf, 4][:200], distance=150)

#print(ttp[0] * 0.2002, data[graf, 3, 0])
plt.plot(lb.data[graf, 4])
for e in ttp[0]:
    plt.plot(e, lb.data[graf, 4][e], 'x')

plt.axvline(int(lb.data[graf, 3]/0.2002))
plt.axvline(ttp[0])
plt.axvline(ttp[0] - int(lb.data[graf, 3]/0.2002)) """

#navečja razlika v nir in peak
""" dif = lb.get_diff(lb.data[:, 4])[0]
plt.plot(dif)
plt.plot(lb.data[:, 4][0] / 35)
p = ss.find_peaks(lb.get_diff(lb.data[:1, 4])[0], height=0, distance=100)[0]
plt.plot(p, dif[p], 'x', color='black') """

#za posamezne label
""" figure, axis = plt.subplots(6)
img = 6

index = 0
for x in lb._image_by_label[:img]:
    for e in x[0]:
        axis[index].plot(e[4], color='blue')
    for e in x[1]:
        axis[index].plot(e[4], color='red')
    for e in x[2]:
        axis[index].plot(e[4], color='green')
    index+=1 """

#za posamezne label 2
""" figure, axis = plt.subplots(2,2)
img = 5

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
