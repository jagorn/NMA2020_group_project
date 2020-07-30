from fran.smz_dataset import get_conditioned_dataset
from fran.smz_plot import *
from hmmlearn import hmm
from matplotlib import pyplot as plt
import pickle

# Parameters
condition_labels = ['right', 'left']
n_compoments = 10
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w', 'gray', 'purple']

# Load Data
dataset = np.load("../models/dataset.npy")
dataset_reduced = np.load("../models/dataset_reduced.npy")
time_bins = np.load("../models/time_bins.npy")
lengths = np.load("../models/lengths.npy")

# Fit model
model = hmm.GaussianHMM(n_components=n_compoments)
model.fit(dataset, lengths)
print("model converged: " + str(model.monitor_.converged))

for condition_label in condition_labels:

    testing_idx = np.load("../models/testing_idx_" + condition_label + ".npy")
    testing_dataset, testing_lengths = get_conditioned_dataset(dataset, lengths, testing_idx)
    testing_dataset_reduced, _ = get_conditioned_dataset(dataset_reduced, lengths, testing_idx)

    # Try Decoding
    [logprob, states] = model.decode(testing_dataset, lengths=testing_lengths, algorithm="viterbi")
    print(logprob)

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
        add_states_2_psth(fig, states[t0:tf], colors, testing_dataset.shape[1])
plt.show()