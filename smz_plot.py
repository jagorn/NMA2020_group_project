from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_psths(dataset, time_bins, title, visual_time, cue_time, feedback_time):
    plt.imshow(dataset, cmap='binary', aspect='auto')
    plt.title(title)
    plt.ylabel('neurons')
    plt.xlabel('time (ms)')

    x_ticks = [0, len(time_bins)-1]
    x_labels = (time_bins[x_ticks] - time_bins[0]) * 1000
    plt.xticks(x_ticks, x_labels.astype(int))

    visual_time_bin = np.where(visual_time < time_bins)[0][0]
    plt.axvline(visual_time_bin, color='r')

    cue_time_bin = np.where(cue_time < time_bins)[0][0]
    plt.axvline(cue_time_bin, color='g')

    feedback_time_bin = np.where(feedback_time < time_bins)[0][0]
    plt.axvline(feedback_time_bin, color='k')

    plt.legend(['visual', 'cue', 'feedback'])


def add_states_2_psth(fig, states, colors, n_neurons):
    # Create a Rectangle patch
    previous_state_change = 0
    state_changes = np.where(np.diff(states))[0]
    for state_change in state_changes:

        state = states[previous_state_change]
        c = colors[state]

        state_dt = state_change - previous_state_change
        rect = patches.Rectangle((previous_state_change, 0), state_dt, n_neurons, edgecolor='none', facecolor=c, alpha=0.2)
        fig.add_patch(rect)
        previous_state_change = state_change

    state = states[previous_state_change]
    c = colors[state]

    state_dt = len(states) - previous_state_change
    rect = patches.Rectangle((previous_state_change, 0), state_dt, n_neurons, edgecolor='none', facecolor=c, alpha=0.2)
    fig.add_patch(rect)



