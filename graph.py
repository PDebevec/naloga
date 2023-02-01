import numpy as np
import lib as lb
import matplotlib.pyplot as plt
import scipy.signal as ss

data = np.load(open('data.npy', 'rb'), allow_pickle=True)

seperate_img = lb.get_image(data)

graf = 0

ttp = ss.find_peaks(data[:, 4][graf][:200], distance=150)
print(ttp[0] * 0.2002, data[:, 3][graf])
plt.plot(data[:, 4][graf])
for e in ttp[0]:
    plt.plot(e, data[:, 4][graf][e], 'x')

plt.axvline(int(data[:, 3][graf]/0.2002))
plt.axvline(ttp[0])
plt.axvline(ttp[0] - int(data[:, 3][graf]/0.2002))

fiture, axis = plt.subplots(3,2)
for i,e in enumerate(seperate_img[:3]):
    for k in e[:, 4]:
        axis[i, 0].plot(k)
    for k in e[:, 6]:
        axis[i, 1].plot(k)
plt.show()