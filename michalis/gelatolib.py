import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


""" STEP 1: FIND THE FILES OF INTEREST """

# Path to the extracted data of interest
mouse = ("/home/michalis/Data/NMA material/Steinmetz dataset/9598406/"
 	    "spikeAndBehavioralData/allData/Cori_2016-12-14")

# Info about channels <--> brain areas
f_areas = "channels.brainLocation.tsv"
with open(f"{mouse}/{f_areas}") as file: # open file as a single string
    lines = file.read().splitlines() # split new lines (string --> list of strings)

# Quality of recordings for a given active neuron
f_quality = "clusters._phy_annotation.npy"
# Which neuron was detected by which channel
f_channels = "clusters.peakChannel.npy"
# Active neurons indices
f_clusters = "spikes.clusters.npy"
# Time of spike
f_times = "spikes.times.npy"


"""STEP 2: LOAD FILES, CLEAN UNECESSARY INFORMATION, ORGANIZE DATA """

# A list where index --> channel,  value --> brain area
areas_from_channels = [line.split('\t')[-1] for line in lines[1:]]
# 1D numpy array where index --> neuron, value --> channel where it is recorded from
channels = np.load(f"{mouse}/{f_channels}")
# 1D numpy array where index --> neuron and value --> recording quality score (good >= 2)
quality = np.load(f"{mouse}/{f_quality}") # index --> cluster (neuron)
# 1D numpy array where index --> spike event and value --> neuron index
clusters = np.load(f"{mouse}/{f_clusters}")
# 1D numpy array where index --> spike event and value --> spike time
times = np.load(f"{mouse}/{f_times}")
# 2d numpy array containing all spiketimes
spiketimes = np.hstack((clusters, times))


""" STEP 3: CREATE SOME USEFUL FUNCTIONS """

def sort_by_area():
    """
    Returns a dict where keys --> brain areas and values --> list of neurons.
    Neurons with bad quality scores (less than 2) are exluded.
    """
    # Create a list where index --> neuron and value --> area
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


def get_area(area=None):
    """
    Returns the spiketimes of single brain area
    
    Parameters
    ----------
    area: str

    Returns
    -------
    area_spikes: 2d np array where 1st column --> area-specific neuron IDs and
                 2nd column --> spike times
    """
    # Set global vars that can be accessed outside of function for debuging
    global sorted_areas, chosen_area, mask
    # Create a dicionary where keys --> brain areas and values --> list of neurons
    sorted_areas = sort_by_area()
    # Choose area of interest
    chosen_area = sorted_areas[area]
    # Create a mask to keep only rows of interest
    mask = np.isin(spiketimes[:,0], chosen_area)
    # Fetch only desired spiketimes
    area_spikes = spiketimes[mask]
    # Find number of neurons
    N_neurons = len(chosen_area)
    
    return area_spikes


def get_spikes(spiketimes=None, t1=None, t2=None):
    """
    Reurns the spiketimes between two time points

    Parameters
    ----------
    t1 : float
        first time point in seconds
    t2 : float
        second time point in seconds
    spikes: 2d np array
        an array of spiketimes

    Returns
    -------
    timed_spikes: 2d np array of spiketimes between t1 and t2

    """
    indices = np.where((spiketimes[:,1] > t1) & (spiketimes[:,1] < t2))
    timed_spikes = spiketimes[indices]
    return timed_spikes
    

def bin_spikes(spiketimes=None, t1=None, t2=None, dt=None):
    neurons_IDs = set(spiketimes[:, 0])
    n_neurons = len(neurons_IDs)

    time_steps = np.arange(t1, t2+dt, dt)
    time_bins = time_steps[1:]
    n_time_bins = len(time_bins)

    spike_counts = np.zeros((n_neurons, n_time_bins))
    
    for i, neuron_id in enumerate(neurons_IDs):
        # for each neuron, retrieve its spikes
        mask = spiketimes[:, 0] == neuron_id
        neuron_spikes = spiketimes[mask][:, 1]


        # generate an histogram of spike counts
        neuron_spike_counts, _ = np.histogram(neuron_spikes, time_steps)
        spike_counts[i, :] = neuron_spike_counts
        
    return spike_counts


""" STEP 4: RUN ANALYSIS """

# sorted_areas = sort_by_area()
mos = get_area('MOs')


# # Visualise spikes
# sns.set()
# plt.plot(ca3[:, 1], ca3[:, 0],  'o', ms=0.5)
# plt.show()

