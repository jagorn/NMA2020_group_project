import ssm
from hmmlearn import hmm
from gelatolib import *
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from get_clean_trials import *


# Analysis parameters
mouse = "Cori_2016-12-14"
area = 'MOs'
pre_stim = 0.3    # seconds
post_stim = 1.5    # seconds
dt = 0.05    # bin size (seconds)
N_PCs = 10  # number of principal components for PCA
N_states = 3   # number of states for HMM


# Analysis options
plot_3d = 0    # True --> shows a 3D plot of first 3 PCs
fit_hmm = 1    # True --> fits a HMM model using hmmlearn


# Data preprocessing
collect_mouse_data(mouse)
trials = extract_clean_trials(mouse)
# Find important timepoints
start_times = trials['interval'][:,0]
end_times = trials['interval'][:,1]
vstim_times = trials['visStim_times'].flatten()
cue_times = trials['cue_times'].flatten()
# Find choises
choices = trials['choice'].flatten()
# Stimulus properties
contrast_R = trials['right_contrast'].flatten()
contrast_L = trials['left_contrast'].flatten()

info_table = np.vstack([start_times, vstim_times, cue_times, end_times, 
                        contrast_L, contrast_R, choices]).T


# Collect and bin spikes
area_spikes = get_area(area)
t_stim = vstim_times[74]
binned_spikes = bin_spikes(area_spikes, t_stim-pre_stim, t_stim+post_stim, dt)


# Perform PCA
X = binned_spikes.T
pca = PCA(n_components=N_PCs)
principalComponents = pca.fit_transform(X)
var_list = pca.explained_variance_ratio_
variance = np.sum(var_list)
print(f"\nVariance explained: {variance:.2f}")
dataset = principalComponents
    
    
if plot_3d:
    # Create 3D plot of first 3 PCs
    plt.figure()
    ax = plt.axes(projection='3d')
    xline = principalComponents[:,0]
    yline = principalComponents[:,1]
    zline = principalComponents[:,2]
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.plot3D(xline, yline, zline, 'o-', ms=1)
    plt.show()


if fit_hmm:
    model = hmm.GaussianHMM(n_components=N_states, n_iter=100)
    model.fit(dataset)
    print(model.monitor_)
    posteriors = model.predict_proba(dataset)
    logprob, states = model.decode(dataset)
    plt.figure(figsize=[9,5])
    for s in range(N_states):
        plt.plot(posteriors[:, s], label="State %d" % s)
    plt.suptitle('Posterior probability of latent states')
    plt.xlabel(f'time bin')
    plt.ylabel('probability')
    plt.legend()
    
    # plt.imshow(states[None,:], aspect="auto")
    plt.show()
    
    
