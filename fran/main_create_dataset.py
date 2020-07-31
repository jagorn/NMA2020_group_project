from fran.smz_load import *
from fran.smz_trials import *
from fran.smz_pca import reduce_dimensionality

# Experimental Parameters
recording_name = 'Cori_2016-12-14'
brain_region = 'MOs'
neuron_min_score = 2

# Dataset Parameters
bin_dt = 0.05  # seconds
pre_cue_dt = 0.2 # seconds
post_resp_dt = 0.2 # seconds
explained_variance = .9

# Choose trials
trials = extract_clean_trials(recording_name)

# compute the edges of each trial
abs_visual_times = trials['visStim_times']
abs_cue_times = trials['cue_times']
abs_feedback_times = trials['feedback_times']

t0s = abs_cue_times - pre_cue_dt
tfs = abs_feedback_times + post_resp_dt
duration = np.median(tfs - t0s)

# Save visual cue and feedback times
visual_times = abs_visual_times - t0s
cue_times = abs_cue_times - t0s
feedback_times = abs_feedback_times - t0s

np.save("../models/visual_times", visual_times)
np.save("../models/cue_times", cue_times)
np.save("../models/feedback_times", feedback_times)

print('data loaded')

# generate the spike count histograms
[dataset, lengths, time_bins] = generate_spike_counts(recording_name, brain_region, neuron_min_score, bin_dt, t0s, duration)
np.save("../models/dataset", dataset)
np.save("../models/lengths", lengths)
np.save("../models/time_bins", time_bins)

print('dataset created')

# reduce dimensionality
dataset_reduced = reduce_dimensionality(dataset, explained_variance)
np.save("../models/dataset_reduced", dataset_reduced)

print('dimensionality reduction done')

# split Trials into training and testing set
conditioned_left = np.where(trials['choice'] == -1)[0]
conditioned_right = np.where(trials['choice'] == 1)[0]

# First for left trials
n_trials_left = len(conditioned_left)
n_testing_set_left = n_trials_left // 10
permutation_left = np.random.permutation(n_trials_left)
testing_idx_left = permutation_left <= n_testing_set_left
testing_left = conditioned_left[testing_idx_left]
training_left = conditioned_left[~testing_idx_left]

np.save("../models/testing_idx_left", testing_left)
np.save("../models/training_idx_left", training_left)

# Then for right trials
n_trials_right = len(conditioned_right)
n_testing_set_right = n_trials_right // 10
permutation_right = np.random.permutation(n_trials_right)
testing_idx_right = permutation_right <= n_testing_set_right
testing_right = conditioned_right[testing_idx_right]
training_right = conditioned_right[~testing_idx_right]

np.save("../models/testing_idx_right", testing_right)
np.save("../models/training_idx_right", training_right)

print('split in training and testing sets done')

