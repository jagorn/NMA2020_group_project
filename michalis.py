import numpy as np
import matplotlib.pyplot as plt

cori = "/home/michalis/Data/NMA material/Steinmetz dataset/9598406/spikeAndBehavioralData/allData/Cori_2016-12-14"
times = np.load(f"{cori}/spikes.times.npy")
clusters = np.load(f"{cori}/spikes.clusters.npy")


plt.plot(times[0:10000] ,clusters[0:10000], 'o', ms=1)
plt.show()
