import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(2020)


def bin_spikes(spiketimes=None, t1=None, t2=None, dt=None):
    neurons_IDs = set(spiketimes[:, 0])
    n_neurons = len(neurons_IDs)

    time_steps = np.arange(t1, t2+dt, dt)
    time_bins = time_steps[1:]
    n_time_bins = len(time_bins)

    spike_counts = np.zeros((n_neurons, n_time_bins))
    
    for i, neuron_id in enumerate(neurons_IDs):
        # for each neuron, retrieve its spikes
        mask = spiketimes[:, 0] == neuron_id
        neuron_spikes = spiketimes[mask][:, 1]


        # generate an histogram of spike counts
        neuron_spike_counts, _ = np.histogram(neuron_spikes, time_steps)
        spike_counts[i, :] = neuron_spike_counts
        
    return spike_counts



"""Create a fake dataset to mimic steinmetz"""

clusters = np.hstack([np.arange(20), np.arange(20)]) # active neuron IDs
times = np.random.randint(100, size=(40)) # random numbers of spike monments
all_spiketimes = np.vstack((clusters, times)).T # column1 --> neurons, column2-->spike times


# """Isolate CA3 neurons """

# ca3_indices = np.arange(0,20,2) # let them all be even numbers for simplicity
# bool_mask = np.isin(all_spiketimes[:,0], ca3_indices) # used to filter ca3 indices
# ca3_spiketimes = all_spiketimes[bool_mask] # column1 --> ca3 neurons, column2-->spike time
# ca3_bins = bin_spikes(ca3_spiketimes, 0, 100, 20)



"""Visualize spiketimes"""
binned_spikes = bin_spikes(all_spiketimes, 0, 100, 20)
sns.set()

fig, axes = plt.subplots(1,2, figsize=(12,5))
ax1, ax2 = axes[0], axes[1]
ax1.plot(all_spiketimes[:, 1], all_spiketimes[:, 0], 'o', ms=8, label='spikes')
# ax.plot(ca3_spiketimes[:, 1], ca3_spiketimes[:, 0], 'rx', ms=8, label='ca3')
# ax.set_title("Even IDs correctly marked as CA3 neurons")
ax1.set_yticks(np.arange(0, 25, step=5))
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('neuron ID')

sns.heatmap(binned_spikes, ax=ax2, cmap="YlGnBu_r")
ax2.invert_yaxis()
ax2.set_xlabel('time bins (20 ms)')
# for t,n in zip(all_spiketimes[:, 1], all_spiketimes[:, 0]):
#     ax.annotate(f'{n}', xy=(t+2,n+0.1))

# plt.show()
