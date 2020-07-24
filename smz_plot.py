from matplotlib import pyplot as plt
import numpy as np


def plot_dataset(dataset, time_bins, title, visual_time, cue_time):
    plt.imshow(dataset)
    plt.title(title)
    plt.ylabel('neurons')
    plt.xlabel('time (ms)')

    x_ticks = range(0, len(time_bins), 20)
    x_labels = (time_bins[x_ticks] - time_bins[0]) * 1000
    plt.xticks(x_ticks, x_labels.astype(int))

    visual_time_bin = np.where(visual_time < time_bins)[0][0]
    plt.axvline(visual_time_bin, color='r')

    cue_time_bin = np.where(cue_time < time_bins)[0][0]
    plt.axvline(cue_time_bin, color='g')

    plt.legend(['visual', 'cue'])



