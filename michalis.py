import matplotlib.pyplot as plt
import numpy as np


# Path to the extracted data of interest:
mouse = ("/home/michalis/Data/NMA material/Steinmetz dataset/9598406/"
	    "spikeAndBehavioralData/allData/Cori_2016-12-14")

# Info about channels <-> brain areas:
f_areas = "channels.brainLocation.tsv"
with open(f"{mouse}/{f_areas}") as file:    # text file, np.load does not work
    lines = file.read().splitlines()

# Spike indices for the rasterplot:
f_clusters = "spikes.clusters.npy"

# Spike times:
f_times = "spikes.times.npy"

# Quality of recordings for a given active neuron, good >= 2
f_quality = "clusters._phy_annotation.npy"

# Which neuron was detected by which channel:
f_channels = "clusters.peakChannel.npy"

# Loading data:
areas = [line.split('\t') for line in lines]    
clusters = np.load(f"{mouse}/{f_clusters}")
times = np.load(f"{mouse}/{f_times}")
quality = np.load(f"{mouse}/{f_quality}")
channels = np.load(f"{mouse}/{f_channels}")


# TODO: clean data using quality 
# TODO: find the corresponding brain area for each cluster using areas and channels


# Visualise some of the spikes using a rasterplot
plt.plot(times[0:10000] ,clusters[0:10000], 'o', ms=1)
plt.show()
