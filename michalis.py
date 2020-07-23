import matplotlib.pyplot as plt
import numpy as np
from numba import jit


# STEP 1: FIND THE FILES OF INTEREST ..........................................
# Path to the extracted data of interest
mouse = ("/home/michalis/Data/NMA material/Steinmetz dataset/9598406/"
	    "spikeAndBehavioralData/allData/Cori_2016-12-14")

# Info about channels <-> brain areas
f_areas = "channels.brainLocation.tsv"
with open(f"{mouse}/{f_areas}") as file: # open file as a single string
    lines = file.read().splitlines() # split new lines (string -> list of strings)

# Quality of recordings for a given active neuron
f_quality = "clusters._phy_annotation.npy"
# Which neuron was detected by which channel
f_channels = "clusters.peakChannel.npy"
# Active neurons indices
f_clusters = "spikes.clusters.npy"
# Time of spike
f_times = "spikes.times.npy"


# STEP 2: LOAD FILES AND CLEAN UNECESSARY INFORMATION .........................
# A list where index -> channel,  value -> brain area
areas_from_channels = [line.split('\t')[-1] for line in lines[1:]]
# 1D numpy array where index -> neuron, value -> channel where it is recorded from
channels = np.load(f"{mouse}/{f_channels}")
# 1D numpy array where index -> neuron and value -> recording quality score (good >= 2)
quality = np.load(f"{mouse}/{f_quality}") # index -> cluster (neuron)
# 1D numpy array where index -> spike event and value -> neuron index
clusters = np.load(f"{mouse}/{f_clusters}")
# 1D numpy array where index -> spike event and value -> spike time
times = np.load(f"{mouse}/{f_times}")


# STEP 3: CREATE SOME USEFUL FUNCTIONS .......................................
def sort_and_clean():
    """
    Returns a dict where keys -> brain areas and values -> list of neurons.
    Neurons with bad quality scores (less than 2) are exluded.
    """
    # Create a list where index -> neuron and value -> area
    matched = [areas_from_channels[int(c)] for c in channels]
    # Find the indices (aka neurons) where they have a score < 2
    bad_indices = [i for i, score in enumerate(quality) if score[0] < 2]
    # Create a dictionary to sort neurons according to areas
    d = {}
    for index, area in enumerate(matched): # Iterate index and value together
        # Discard bad recordings
        if index not in bad_indices:
            # If the area is already a key then append this neuron index
            if area in d.keys():
                d[area].append(index)
            # Else create a new key for a single element list
            else:
                d[area] = [index]
    return d


# STEP 3: RUN ANALYSIS ........................................................
# Create a dicionary where keys -> brain areas and values -> list of neurons
sorted_areas = sort_and_clean()
# Choose area of interest
ca3 = sorted_areas['CA3']
# Merge spiketimes into a single 2d array
all_spiketimes = np.hstack((clusters, times))
# Create an mask to keep only rows of interest
ca3_filter = np.isin(all_spiketimes[:,0], ca3)
# Fetch only desired spiketimes
ca3_spiketimes = all_spiketimes[ca3_filter]


# Visualise spikes
plt.plot(ca3_spiketimes[:, 1], ca3_spiketimes[:, 0],  'o', ms=0.5)
plt.show()
