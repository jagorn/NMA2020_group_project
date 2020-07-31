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


with open("../models/hmm_left_" + str(n_compoments) + ".pkl", "rb") as file:
    hmm_left = pickle.load(file)

with open("../models/hmm_right_" + str(n_compoments) + ".pkl", "rb") as file:
    hmm_right = pickle.load(file)


for condition_label in condition_labels:

    testing_idx = np.load("../models/testing_idx_" + condition_label + ".npy")
    testing_dataset, testing_lengths = get_conditioned_dataset(dataset, lengths, testing_idx)
    testing_dataset_reduced, _ = get_conditioned_dataset(dataset_reduced, lengths, testing_idx)

    trials_ends = np.cumsum(testing_lengths)
    trials_starts = trials_ends - testing_lengths

    print('condition = ' + condition_label)
    for i in range(len(trials_starts)):

        t0 = trials_starts[i]
        tf = trials_ends[i]

        # Try Decoding
        [logprob_left, states] = hmm_left.decode(testing_dataset_reduced[t0:tf, :])
        [logprob_right, states] = hmm_right.decode(testing_dataset_reduced[t0:tf, :])

        if logprob_left > logprob_right:
            print("choose left (logprob difference = " + str(logprob_left - logprob_right) + ")")
        else:
            print("choose right (logprob difference = " + str(logprob_right - logprob_left) + ")")



