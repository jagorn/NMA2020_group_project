import numpy as np


def get_conditioned_dataset(dataset, lengths, conditioned_trials):
    conditioned_lengths = lengths[conditioned_trials]
    trials_ends = np.cumsum(lengths)[conditioned_trials]
    trials_starts = trials_ends - conditioned_lengths

    idxs = []
    for i in range(len(trials_ends)):
        idxs.append(np.arange(trials_starts[i], trials_ends[i]))
    idxs = np.concatenate(idxs)
    conditioned_dataset = dataset[idxs, :]

    return conditioned_dataset, conditioned_lengths
