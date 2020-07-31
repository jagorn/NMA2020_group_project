%-------
% HMM

% Modified from Mazzucato's demo2_HMM_Simple.m in
% contamineuro_2019_spiking_net by Jiyun Shin
%-------

% Add Mazzucato_HMM folder to the path 

addpath('C:\Users\Larkum_Practical_02\Documents\GitHub\NMA2020_group_project\Mazzucato_HMM')
%% Data preparation
% change the filepath to your data folder
filepath = 'C:\Users\Larkum_Practical_02\Documents\GitHub\NMA2020_group_project\data\Cori_2016-12-14';

% Load data
data = loadSession(filepath);

% Choose the brain area
area = 'MOs';
%feedback = 1; % hit trials
feedback = []; % all trials

% Reformat data into the 'spikes' structure array with dimension [ntrials,nunits] and field .spk containing spike times as columns array (in seconds)
% and create 'win_train' array with dimensions [ntrials, 2], where each row is the [start, end] times for each trial (spike times in 'spikes' must be consistently aligned with 'win_train')
% -0.5 s to 2 s from the visual stim onset
[win_train, total_spikes] = OrganizeHMMInput (data, area,feedback);
[ntrials, gnunits]=size(total_spikes);


%% HMM PARAMETERS

HmmParam=struct();
% NUMBER OF HIDDEN STATES
HmmParam.VarStates=10; % number of hidden states
%--------------------
% HmmParam.AdjustT=0.1; % interval to skip at trial start to avoid canonical choice of 1st state in matlab
HmmParam.BinSize=0.005;%0.002; % time step (s) of Markov chain
HmmParam.MinDur=0.05;   % min duration of an admissible state (s) in HMM DECODING
HmmParam.MinP=0.8;      % pstate>MinP for an admissible state in HMM ADMISSIBLE STATES
HmmParam.NumSteps=1;%    % 10 number of fits at fixed parameters to avoid non-convexity
HmmParam.NumRuns=100;%     % 50% % number of times we iterate hmmtrain over all trials


%% HMM fitting

% transform spike times into observation sequence
[sequence, ~]=hmm.fun_HMM_binning(total_spikes,HmmParam,win_train); % ID of cells fired at that bin
% train HMM
tic
hmm_bestfit=hmm.fun_HMM_training_NOPARFOR(sequence,gnunits,HmmParam);
fprintf('HMM fit with %d states\n',HmmParam.VarStates);
toc 

%% HMM decoding for all trials

% estimate posterior probabilities
hmm_results=hmm.fun_HMM_decoding(total_spikes,hmm_bestfit,HmmParam,win_train);
% HMM ADMISSIBLE STATES -> state sequences
hmm_postfit=hmm.fun_HMM_postfit(total_spikes,hmm_results,HmmParam,win_train);

% OUTPUT OF HMM FIT
%
% Important variables:
%     hmm_bestfit.tpm: K x K transition probability matrix, where K is the number of hidden states
%     hmm_bestfit.epm: K x (nunits+1) emission probability matrix, where K is the number of hidden states, the (n+1)-th column represents the probability of silence - you can safely drop it
%     hmm_bestfit.LLtrain: -2*loglikelihood of the data 
% 
%     hmm_results(i_trial).pStates: array of dim [K,time] with posterior probabilities of each state in trial i_trial
%     hmm_results(i_trial).rates: array of dim [K,nunits] with local estimate of emissions (i.e., firing rates in each state) conditioned on observations in trial i_trial 
%     hmm_results(i_trial).Logpseq: -2*loglikelihood from local observations in trial i_trial
%     
%     hmm_postfit(i_trial).sequence: array of dimension [4,nseq] where columns represent detected states (intervals with prob(state)>0.8), in the order they appear in trial
%         i_trial, and rows represent state [onset,offset,duration,label].

%% PLOTS
%hmmdir=fullfile('data','hmm'); 
%if ~exist(hmmdir,'dir'); mkdir(hmmdir); end
colors=aux.distinguishable_colors(max(HmmParam.VarStates,4));
% plot tpm and epm
hmm.plot_tpm_epm;

% plot first 10 trials
hmm.plot_trials;










