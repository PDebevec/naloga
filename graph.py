import numpy as np
import lib as lb
import matplotlib.pyplot as plt
import scipy.signal as ss

""" for img in range(1):
    plt.plot(lb.data[img, 4, 0][:300])
    am = np.argmax(lb.data[img, 4, 0][:300])
    plt.plot(am, lb.data[img, 4, 0][am], 'x')
    ttp = int(am *0.1)
    plt.plot(ttp, lb.data[img, 4, 0][ttp], 'x')
    plt.plot(int(lb.data[img, 3, 0]/0.2002), lb.data[img, 4, 0][int(lb.data[img, 3, 0]/0.2002)], 'x') """

""" graf = 0
ttp = ss.find_peaks(lb.data[graf, 4, 0][:200], distance=150)

#print(ttp[0] * 0.2002, data[graf, 3, 0])
plt.plot(lb.data[graf, 4, 0])
for e in ttp[0]:
    plt.plot(e, lb.data[graf, 4, 0][e], 'x')

plt.axvline(int(lb.data[graf, 3, 0]/0.2002))
plt.axvline(ttp[0])
plt.axvline(ttp[0] - int(lb.data[graf, 3, 0]/0.2002)) """

""" dif = lb.get_diff(lb.data[:, 4, 0])[0]
plt.plot(dif)
plt.plot(lb.data[:, 4, 0][0] / 35)
p = ss.find_peaks(lb.get_diff(lb.data[:1, 4, 0])[0], height=0, distance=100)[0]
plt.plot(p, dif[p], 'x', color='black') """

""" figure, axis = plt.subplots(6)
img = 6

index = 0
for x in lb._separated_images[:img]:
    for e in x[0]:
        for k in e[4]:
            axis[index].plot(k, color='blue')
    for e in x[1]:
        for k in e[4]:
            axis[index].plot(k, color='red')
    for e in x[2]:
        for k in e[4]:
            axis[index].plot(k, color='green')
    index+=1 """

""" figure, axis = plt.subplots(2,2)
img = 5

for e in lb._separated_images[img][2]:
    for k in e[4]:
        axis[0, 0].plot(k, color='green')
        axis[1, 0].plot(k, color='green')
for e in lb._separated_images[img][1]:
    for k in e[4]:
        axis[0, 1].plot(k, color='red')
for e in lb._separated_images[img][0]:
    for k in e[4]:
        axis[1, 1].plot(k, color='blue') """

""" fiture, axis = plt.subplots(3,2)
for i,e in enumerate(lb._separated_data[:3]):
    for k in e[:, 4, 0]:
        axis[i, 0].plot(k)
    for k in e[:, 6, 0]:
        axis[i, 1].plot(k) """

plt.show()
