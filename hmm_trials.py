
from smz_load import *
from get_clean_trials import *
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from hmmlearn import hmm


# Get clenaed trial intervals
recording_name = 'Cori_2016-12-14'
clean_trials = clean_trials (recording_name)


# Set parameters
trialNum = 1 # start from 0
trial_onset = clean_trials['interval'][trialNum,0]
trial_offset = clean_trials['interval'][trialNum,1]
dt = 0.05 # binsize, seconds
# t0 = clean_trial_intervals[trialNum,0] # onset, seconds
# tf = clean_trial_intervals[trialNum,1] # offset, seconds
minimum_grade = 2
brain_region = 'MOs'
[dataset, time_bins] = generate_spike_counts(recording_name, brain_region, minimum_grade, dt, t0=trial_onset, tf = trial_offset)


# Create a hmm model
n_compoments = 3
model = hmm.GaussianHMM(n_components=n_compoments, covariance_type="full", n_iter=100)
model.fit(dataset.T)
[score, states] = model.decode(dataset.T)


# Create a Rectangle patch
plt.figure()
plt.plot(states)
# plt.show()

# for state in states
# rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
# ax.add_patch(rect)
# pass
#



plt.figure()
plt.imshow(dataset)
plt.show()