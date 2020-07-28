
from smz_load import *
from smz_plot import *
from get_clean_trials import *
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from hmmlearn import hmm


# Get clenaed trial intervals
recording_name = 'Cori_2016-12-14'
clean_trials = extract_clean_trials (recording_name)


# Set parameters
# trialNum = 1 # start from 0
# trial_onset = clean_trials['interval'][trialNum,0]
# trial_offset = clean_trials['interval'][trialNum,1]
dt = 0.05 # binsize, seconds
# t0 = clean_trial_intervals[trialNum,0] # onset, seconds
# tf = clean_trial_intervals[trialNum,1] # offset, seconds
minimum_grade = 2
brain_region = 'MOs'



n_compoments = 3
colors = ['r', 'g', 'b', 'y', 'm', 'c']

plt.figure()
n_trials = 5
for trial in range(n_trials):
    if clean_trials['feedback_types'][trial] == 1: # Hit trials
        fig = plt.subplot(1, n_trials, trial+1)
    
        visual_time = clean_trials['visStim_times'][trial]
        cue_time = clean_trials['cue_times'][trial]
        trial_interval = clean_trials['interval'][trial]
        [dataset, time_bins] = generate_spike_counts(recording_name, brain_region, minimum_grade, dt, trial_interval[0], trial_interval[1])
        (n_neurons, n_bins) = dataset.shape
    
        # Create a hmm model
        # dataset = (dataset > 0).astype(int)  # Conversion to binary for multinomial HMM
        model = hmm.GaussianHMM(n_components=n_compoments, n_iter=1000)
        model.fit(dataset.T)
        [logprob, states] = model.decode(dataset.T)
    
        title = brain_region + ' trial#' + str(trial)
        plot_psths(dataset, time_bins, title, visual_time, cue_time)
        add_states_2_psth(fig, states, colors, n_neurons)

plt.show()




trial = 1
visual_time = clean_trials['visStim_times'][trial]
cue_time = clean_trials['cue_times'][trial]
trial_interval = clean_trials['interval'][trial]
[dataset, time_bins] = generate_spike_counts(recording_name, brain_region, minimum_grade, dt, trial_interval[0], trial_interval[1])
(n_neurons, n_bins) = dataset.shape
    
        # Create a hmm model
        # dataset = (dataset > 0).astype(int)  # Conversion to binary for multinomial HMM
model = hmm.GaussianHMM(n_components=n_compoments, n_iter=1000)
model.fit(dataset.T)
[logprob, states] = model.decode(dataset.T)
