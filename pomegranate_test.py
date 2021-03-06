from smz_load import *
from smz_plot import *
from pomegranate import *

# Data Parameters
recording_name = 'Cori_2016-12-14'
brain_region = 'MOs'
dt = 0.02  # seconds
minimum_grade = 2

# Model parameters
compoments = 3
colors = ['r', 'g', 'b', 'y', 'm', 'c']

# plt.figure()
n_trials = 1
for trial in range(n_trials):
    # fig = plt.subplot(1, n_trials, trial+1)

    visual_time = load_visual_stim_times(recording_name)[trial]
    cue_time = load_cue_times(recording_name)[trial]
    trial_interval = load_trial_intervals(recording_name, trial)
    [dataset, time_bins] = generate_spike_counts(recording_name, brain_region, minimum_grade, dt, trial_interval[0], trial_interval[1])
    (n_neurons, n_bins) = dataset.shape

    # Create a hmm model
    dataset = (dataset > 0).astype(int)
    model = HiddenMarkovModel.from_samples(BernoulliDistribution, n_components=compoments, X=[dataset.T])
    # [logprob, states] = model.viterbi(dataset)


    
    
#     model.fit(dataset.T)
#     [logprob, states] = model.decode(dataset.T)

#     title = brain_region + ' trial#' + str(trial)
#     plot_psths(dataset, time_bins, title, visual_time, cue_time)
#     add_states_2_psth(fig, states, colors, n_neurons)

# plt.show()
