import numpy as np
import pathlib
import os.path
import csv

""" FILE NAMES """
# Quality of recordings for a given active neuron
f_quality = "clusters._phy_annotation.npy"
# Which neuron was detected by which channel:
f_channels = "clusters.peakChannel.npy"
# Active neurons indices:
f_clusters = "spikes.clusters.npy"
# Time of spike:
f_times = "spikes.times.npy"
# Info about channels <-> brain areas:
f_areas = "channels.brainLocation.tsv"
# Location of the project
project_path = pathlib.Path().absolute()


""" LOADING FUNCTIONS """
def load_neuron_regions(recording_name):
    data_path = os.path.join(project_path, 'data', recording_name)

    # Load info about cluster's channel
    channels_file = os.path.join(data_path, f_channels)
    neurons_2_channels = np.load(channels_file)

    # Load info about channel's regions
    areas_file = os.path.join(data_path, f_areas)
    reader = csv.reader(open(areas_file, "r"), delimiter="\t")
    regions_2_channels = np.array(list(reader))[:, 3]

    # Retrieve the cluter's region
    neuron_2_regions = regions_2_channels[neurons_2_channels]
    return neuron_2_regions


def load_neuron_grades(recording_name):
    data_path = os.path.join(project_path, 'data', recording_name)
    grades_file = os.path.join(data_path, f_quality)
    neuron_2_grades = np.load(grades_file)
    return neuron_2_grades

def load_spike_times(recording_name):
    data_path = os.path.join(project_path, 'data', recording_name)
    spikes_file = os.path.join(data_path, f_times)
    spikes = np.load(spikes_file)
    return spikes

def load_spikes_2_neurons(recording_name):
    data_path = os.path.join(project_path, 'data', recording_name)
    neurons_file = os.path.join(data_path, f_clusters)
    spikes_2_neurons = np.load(neurons_file)

def select_neurons_by_region:
    pass

def select_neurons_by_grade:
    pass

def create_my_dataset(recording_name, region, grades):
    neuron_2_regions = load_brain_regions(recording_name)
