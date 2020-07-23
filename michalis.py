import matplotlib.pyplot as plt
import numpy as np

# STEP 1: FIND THE FILES OF INTEREST ..........................................

# Path to the extracted data of interest:
mouse = ("/home/michalis/Data/NMA material/Steinmetz dataset/9598406/"
	    "spikeAndBehavioralData/allData/Cori_2016-12-14")

# Info about channels <-> brain areas:
f_areas = "channels.brainLocation.tsv"
with open(f"{mouse}/{f_areas}") as file: # open file as a single string
    lines = file.read().splitlines() # split new lines (string -> list of strings)

# Quality of recordings for a given active neuron
f_quality = "clusters._phy_annotation.npy"
# Which neuron was detected by which channel:
f_channels = "clusters.peakChannel.npy"
# Active neurons indices:
f_clusters = "spikes.clusters.npy"
# Time of spike:
f_times = "spikes.times.npy"

# STEP 2: LOAD FILES AND CLEAN UNECESSARY DATA ................................

# A list where index = channel,  value = brain area
areas_from_channels = [line.split('\t')[-1] for line in lines[1:]]
# 1D numpy array where index = neuron, value = channel where it is recorded from
channels = np.load(f"{mouse}/{f_channels}")
# 1D numpy array where index = neuron and value = recording quality score (good >= 2)
quality = np.load(f"{mouse}/{f_quality}") # index = cluster (neuron)
# 1D numpy array where index = spike event and value = neuron index
clusters = np.load(f"{mouse}/{f_clusters}")
# 1D numpy array where index = spike event and value = spike time
times = np.load(f"{mouse}/{f_times}")


def clear_data():
    """Remove neurons (clusters) with quality score < 2"""
    pass

def find_area():
    """Match neurons with their corresponding areas"""
    pass



# Visualise some of the spikes using a rasterplot
# plt.plot(times[0:10000] ,clusters[0:10000], 'o', ms=1)
# plt.show()
