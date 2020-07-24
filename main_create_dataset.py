from smz_load import *
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from hmmlearn import hmm

# Load my Data
recording_name =  'Tatum_2017-12-06'  #  'Cori_2016-12-14'
dt = 0.05 # seconds
t0 = 62.9 # seconds
tf = t0 + 5 # seconds
minimum_grade = 2
brain_region = 'MOs'
[dataset, time_bins] = generate_spike_counts(recording_name, brain_region, minimum_grade, dt, t0, tf)


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