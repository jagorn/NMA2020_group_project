from smz_load import *
from matplotlib import pyplot as plt

recording_name = 'Cori_2016-12-14'
dt = 0.05 # seconds
t0 = 0 # seconds
tf = 15 # seconds
minimum_grade = 2
brain_region = 'LS'
dataset = generate_spike_counts(recording_name, brain_region, minimum_grade, dt, t0, tf)


plt.imshow(dataset)
plt.show()