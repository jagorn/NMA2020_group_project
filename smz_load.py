import pathlib
import os.path
import csv
import numpy as np

""" FILE NAMES """
# Quality of recordings for a given active neuron
f_quality = "clusters._phy_annotation.npy"
# Which neuron was detected by which channel:
f_channels = "clusters.peakChannel.npy"
# Active neurons indices:
f_clusters = "spikes.clusters.npy"
# Info about channels <-> brain areas:
f_areas = "channels.brainLocation.tsv"

# Time of spike:
f_times = "spikes.times.npy"
# Times of visual stimulations for each trial
f_visual = "trials.visualStim_times.npy"
# Times of auditory cue for each trial
f_cue = "trials.goCue_times.npy"


# Info about the trial time intervals:
f_trials = "trials.intervals.npy"
# Info about the trials used:
f_trials_idx = "trials.included.npy"

# Location of the project
project_path = pathlib.Path().absolute()


""" LOADING FUNCTIONS """


def load_visual_stim_times(recording_name):
    # Returns a vector of length n_trials. Each value tells the time at which the stimulus was played for a given trial
    data_path = os.path.join(project_path, 'data', recording_name)
    visual_times_file = os.path.join(data_path, f_visual)
    visual_times = np.load(visual_times_file)
    visual_times = visual_times.reshape(len(visual_times))
    return visual_times


def load_cue_times(recording_name):
    # Returns a vector of length n_trials. Each value tells the time at which the cue was played for a given trial
    data_path = os.path.join(project_path, 'data', recording_name)
    cue_times = os.path.join(data_path, f_cue)
    cue_times = np.load(cue_times)
    cue_times = cue_times.reshape(len(cue_times))
    return cue_times


def load_neuron_regions(recording_name):
    # Returns a vector of length n_neurons. Each value tells the brain area of the given neuron
    data_path = os.path.join(project_path, 'data', recording_name)

    # Load info about neuron's channel
    channels_file = os.path.join(data_path, f_channels)
    neurons_2_channels = np.load(channels_file)
    neurons_2_channels = neurons_2_channels.reshape(len(neurons_2_channels))

    # Load info about channel's regions
    areas_file = os.path.join(data_path, f_areas)
    reader = csv.reader(open(areas_file, "r"), delimiter="\t")
    regions_2_channels = np.array(list(reader))[:, 3]
    neurons_2_channels = neurons_2_channels.reshape(len(neurons_2_channels))
    neurons_2_channels = neurons_2_channels.astype(int)

    # Combine the information above to discover which region corresponds to each neuron
    neuron_2_regions = regions_2_channels[neurons_2_channels]
    return neuron_2_regions


def load_neuron_grades(recording_name):
    # Returns a vector of length n_neurons. Each value tells the grade (1=bad, 2=decent, 3=good) of the given neuron
    data_path = os.path.join(project_path, 'data', recording_name)
    grades_file = os.path.join(data_path, f_quality)
    neuron_2_grades = np.load(grades_file)
    neuron_2_grades = neuron_2_grades.reshape(len(neuron_2_grades))
    neuron_2_grades = neuron_2_grades.astype(int)
    return neuron_2_grades


def load_trial_intervals(recording_name, idx=None):
    # Returns a vector of length n_neurons. Each value tells the grade (1=bad, 2=decent, 3=good) of the given neuron
    data_path = os.path.join(project_path, 'data', recording_name)
    trials_file = os.path.join(data_path, f_trials)
    trials_indices_file = os.path.join(data_path, f_trials_idx)

    trials = np.load(trials_file)
    trials_indices = np.load(trials_indices_file)
    selected_trials = clean_trials(trials, trials_indices)

    if idx is not None:
        selected_trials = selected_trials[idx]

    return selected_trials


def load_spike_times(recording_name):
    # Returns a vector of length n_spikes. Each value tells the time at which the spike was recorded (in seconds)
    data_path = os.path.join(project_path, 'data', recording_name)
    spikes_file = os.path.join(data_path, f_times)
    spikes = np.load(spikes_file)
    spikes = spikes.reshape(len(spikes))
    return spikes


def load_spikes_2_neurons(recording_name):
    # Returns a vector of length n_spikes. Each value is the identifier of the neuron to which the spike belongs
    data_path = os.path.join(project_path, 'data', recording_name)
    neurons_file = os.path.join(data_path, f_clusters)
    spikes_2_neurons = np.load(neurons_file)
    spikes_2_neurons = spikes_2_neurons.reshape(len(spikes_2_neurons))
    spikes_2_neurons = spikes_2_neurons.astype(int)
    return spikes_2_neurons


""" DATA-SET FUNCTIONS """


def clean_trials(trial_intervals, included_trials):
    #   Remove trials that were not included
    #   INPUT: trial_intervals  - onset and offset timestamps of a N trials
    #          included_trials - boolean +1 included, 0 not included
    #
    idx = np.where(included_trials == 1)
    idx = idx[0]
    return trial_intervals[idx, :]


def select_neurons_by_region(recording_name, region):
    # Return a vector with the identifiers of all the neurons belonging to a given brain area
    neuron_2_regions = load_neuron_regions(recording_name)
    neuron_indices = np.where(np.char.equal(region, neuron_2_regions))[0]
    return neuron_indices


def select_neurons_by_grade(recording_name, minimum_grade):
    # Return a vector with the identifiers of all the neurons with score >= minimum_grade
    neuron_2_grades = load_neuron_grades(recording_name)
    neuron_indices = np.where(neuron_2_grades > minimum_grade)[0]
    return neuron_indices


def generate_spike_counts(recording_name, brain_region, minimum_grade, dt, t0, tf, firing_rates=False):
    # Selects the neuron belonging to a given brain region, and generates a data-set.
    # Returns: a matrix [n_neurons * n_time_steps] describing how many times each neuron spiked at a given time step
    # Arguments:
    # brain_region: the brain region we want to analyze
    # minimum_grade: the minimum score we accept to include a neuron in the data-set (1=bad, 2=decent, 3=good)
    # t0 = the initial time step (seconds)
    # tf = the final time step (seconds)
    # dt = the length of each time step (seconds)
    # firing_rates = if false, the returned values are spike counts. if true, it is firing rates

    # select the neurons based on brain region and score
    neuron_indices_1 = select_neurons_by_region(recording_name, brain_region)
    neuron_indices_2 = select_neurons_by_grade(recording_name, minimum_grade)
    neuron_indices = np.intersect1d(neuron_indices_1, neuron_indices_2)

    # load spikes and spike mappings
    spikes = load_spike_times(recording_name)
    spikes_2_neurons = load_spikes_2_neurons(recording_name)

    # generate the time steps sequence
    time_steps = np.arange(t0, tf, dt)
    n_time_bins = len(time_steps) - 1
    time_bins = time_steps[1:]

    n_neurons = len(neuron_indices)
    spike_counts = np.zeros((n_neurons, n_time_bins))

    for i, neuron_id in enumerate(neuron_indices):
        # for each neuron, retrieve its spikes
        neuron_spikes = spikes[spikes_2_neurons == neuron_id]

        # generate an histogram of spike counts
        [neuron_spike_counts, _] = np.histogram(neuron_spikes, time_steps)
        if firing_rates:
            neuron_spike_counts /= dt
        spike_counts[i, :] = neuron_spike_counts

    return spike_counts, time_bins
