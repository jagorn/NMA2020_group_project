from fran.smz_dataset import get_conditioned_dataset
from fran.smz_plot import *
from hmmlearn import hmm
from matplotlib import pyplot as plt
import pickle

# Parameters
condition_labels = ['right', 'left']
n_compoments = 3
colors = ['r', 'g', 'b', 'y', 'm', 'c']

# Load Data
dataset = np.load("../models/dataset.npy")
dataset_reduced = np.load("../models/dataset_reduced.npy")
time_bins = np.load("../models/time_bins.npy")
lengths = np.load("../models/lengths.npy")

for condition_label in condition_labels:

    # get training set trials
    training_idx = np.load("../models/training_idx_" + condition_label + ".npy")
    training_dataset, training_lengths = get_conditioned_dataset(dataset, lengths, training_idx)
    training_dataset_reduced, _ = get_conditioned_dataset(dataset_reduced, lengths, training_idx)

    testing_idx = np.load("../models/testing_idx_" + condition_label + ".npy")
    testing_dataset, testing_lengths = get_conditioned_dataset(dataset, lengths, testing_idx)
    testing_dataset_reduced, _ = get_conditioned_dataset(dataset_reduced, lengths, testing_idx)

    # Plot
    visual_times = np.load("../models/visual_times.npy")[testing_idx]
    cue_times = np.load("../models/cue_times.npy")[testing_idx]
    feedback_times = np.load("../models/feedback_times.npy")[testing_idx]

    plt.figure()

    trials_ends = np.cumsum(testing_lengths)
    trials_starts = trials_ends - testing_lengths
    n_trials_plotted = len(trials_starts)

    for i in range(n_trials_plotted):
        fig = plt.subplot(1, n_trials_plotted, i + 1)

        t0 = trials_starts[i]
        tf = trials_ends[i]
        plot_psths(testing_dataset[t0:tf, :].T, time_bins, "psths", visual_times[i], cue_times[i], feedback_times[i])
plt.show()