from smz_load import *
from smz_plot import *
from get_clean_trials import *
from matplotlib import pyplot as plt
from hmm_map_states import *
import ssm
import matplotlib.cm as cm


# Experimental Parameters
recording_name = 'Cori_2016-12-14'
brain_region = 'MOs'
neuron_min_score = 2

# Model Parameters
bin_dt = 0.002  # seconds
pre_stim_dt = 0.5 # seconds
post_resp_dt = 0.5 # seconds
n_compoments = 5
colors = ['r', 'g', 'b', 'y', 'm', 'c']

# Choose trials
trials = extract_clean_trials(recording_name)
conditioned_trials = np.where(trials['choice'] == 0)[0]

# Run the fit
n_trials = 5
for i in range(n_trials, n_trials+1):

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
    dataset = (dataset > 0).astype(int)
    model = ssm.HMM(n_compoments, 2, observations="bernoulli")
    hmm_lls = model.fit(dataset.T, method="em", num_iters=200)
    states = model.most_likely_states(dataset.T)
    ninja = model.filter(dataset.T)

time = range(len(ninja[:, 0]))

plt.plot(time, ninja[:, 0])
plt.plot(time, ninja[:, 1])
plt.plot(time, ninja[:, 2])
plt.plot(time, ninja[:, 3])
plt.plot(time, ninja[:, 4])
plt.show()

# plt.imshow(states[None,:], aspect="auto", cmap=cm.RdYlGn, vmin=0, vmax=len(colors)-1)
# # plt.xlim(0, time_bins)
# # plt.ylabel("$z_{\\mathrm{inferred}}$")
# plt.yticks([])
# plt.xlabel("time")
# # plt.plot(hmm_lls, label="EM")
# # # plt.plot([0, 1000], true_ll * np.ones(2), ':k', label="True")
# # plt.xlabel("EM Iteration")
# # plt.ylabel("Log Probability")
# # plt.legend(loc="lower right")
# plt.show()