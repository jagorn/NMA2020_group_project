from smz_load import *
from smz_plot import *
from get_clean_trials import *
from matplotlib import pyplot as plt
from hmm_map_states import *
import ssm
import matplotlib.cm as cm


# np.random.seed(0)


# Experimental Parameters
recording_name = 'Cori_2016-12-14'
brain_region = 'MOs'
neuron_min_score = 2

# Model Parameters
bin_dt = 0.01  # seconds
pre_stim_dt = 0.5 # seconds
post_resp_dt = 0.5 # seconds
N_states = 3


# Choose trials
trials = extract_clean_trials(recording_name)
conditioned_trials = np.where(trials['choice'] == 1)[0]

    
# Run the fit
n_trials = 1
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
    train_data = (dataset > 0).astype(int).T
    model = ssm.HMM(N_states, n_neurons, observations="poisson")
    hmm_lls = model.fit(train_data, method="em", num_iters=1000)
    posterior = model.filter(train_data)


plt.figure(n_trials, figsize=[9,5])
for s in range(N_states):
    plt.plot(posterior[:, s], label="State %d" % s)
    
plt.suptitle('Posterior probability of latent states')
plt.xlabel(f'time bin ({int(bin_dt*1000)} ms)')
plt.ylabel('probability')

plt.legend()
plt.show()

