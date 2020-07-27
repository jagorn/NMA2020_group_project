from smz_load import *
from smz_plot import *
from get_clean_trials import *
from matplotlib import pyplot as plt
from hmm_map_states import *
from hmmlearn import hmm

# Experimental Parameters
recording_name = 'Cori_2016-12-14'
brain_region = 'ACA'
neuron_min_score = 2

# Model Parameters
bin_dt = 0.05  # seconds
pre_stim_dt = 0.5 # seconds
post_resp_dt = 0.5 # seconds
n_compoments = 5
colors = ['r', 'g', 'b', 'y', 'm', 'c']

# Choose trials
trials = extract_clean_trials(recording_name)
conditioned_trials = np.where(trials['choice'] == 0)[0]

# Run the fit
plt.figure()
n_trials = 5
for i in range(n_trials):
    fig = plt.subplot(1, n_trials, i + 1)

    # Load time pointers for the given trial
    trial = conditioned_trials[i]
    visual_time = trials['visStim_times'][trial]
    cue_time = trials['cue_times'][trial]
    feedback_time = trials['feedback_times'][trial]

    # generate the spike count histograms
    t0 = visual_time - pre_stim_dt
    tf = feedback_time + post_resp_dt
    [dataset, time_bins] = generate_spike_counts(recording_name, brain_region, neuron_min_score, bin_dt, t0, tf)
    (n_neurons, n_bins) = dataset.shape

    # Create a hmm model
    model = hmm.GaussianHMM(n_components=n_compoments, n_iter=1000)
    model.fit(dataset.T)
    [logprob, states] = model.decode(dataset.T)

    # Find the best mapping of the state sequences
    if i == 0:
        states_trial0 = states
    else:
        states = map_states(n_compoments, states_trial0, states)

    # Plot
    title = brain_region + ' trial#' + str(trial)
    plot_psths(dataset, time_bins, title, visual_time, cue_time, feedback_time)
    add_states_2_psth(fig, states, colors, n_neurons)

plt.show()