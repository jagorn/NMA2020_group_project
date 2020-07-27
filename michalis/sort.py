import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2020)


"""Create a fake dataset to mimic steinmetz"""

clusters = np.arange(20) # active neuron IDs
times = np.random.randint(100, size=(20)) # random numbers of spike monments
all_spiketimes = np.vstack((clusters, times)).T # column1 --> neurons, column2-->spike times



"""Isolate CA3 neurons """

ca3_indices = np.arange(0,20,2) # let them all be even numbers for simplicity
bool_mask = np.isin(all_spiketimes[:,0], ca3_indices) # used to filter ca3 indices
ca3_spiketimes = all_spiketimes[bool_mask] # column1 --> ca3 neurons, column2-->spike time



"""Visualize spiketimes"""

fig, ax = plt.subplots()

ax.plot(all_spiketimes[:, 1], all_spiketimes[:, 0], 'o', ms=8, label='spikes')
ax.plot(ca3_spiketimes[:, 1], ca3_spiketimes[:, 0], 'rx', ms=8, label='ca3')

ax.set_title("Even IDs correctly marked as CA3 neurons")
ax.set_yticks(np.arange(0, 25, step=5))
ax.set_xlabel('time (ms)')
ax.set_ylabel('neuron ID')
plt.legend()

for t,n in zip(all_spiketimes[:, 1], all_spiketimes[:, 0]):
    ax.annotate(f'{n}', xy=(t+2,n+0.1))

plt.show()
