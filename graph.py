import numpy as np
import lib as lb
import matplotlib.pyplot as plt
import scipy.signal as ss

data = np.load(open('data.npy', 'rb'), allow_pickle=True)

#print(data[:, 0, 0])
seperate_img = lb.get_image(data)
#print(seperate_img[0][:, 4])

""" graf = 0

ttp = ss.find_peaks(data[graf, 4, 0][:200], distance=150)

#print(ttp[0] * 0.2002, data[graf, 3, 0])
plt.plot(data[graf, 4, 0])
for e in ttp[0]:
    plt.plot(e, data[graf, 4, 0][e], 'x')

plt.axvline(int(data[graf, 3, 0]/0.2002))
plt.axvline(ttp[0])
plt.axvline(ttp[0] - int(data[graf, 3, 0]/0.2002)) """

""" fiture, axis = plt.subplots(3,2)
for i,e in enumerate(seperate_img[:3]):
    for k in e[:, 4, 0]:
        axis[i, 0].plot(k)
    for k in e[:, 6, 0]:
        axis[i, 1].plot(k) """
plt.show()
