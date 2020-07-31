function spikes_in_trials = extract_spikes_in_trials(trial_intervals, spiketimes, clusters)
% Extract spiketimes and unit id (clusters) occured within the trial
% window.
% INPUT : trial_intervals (Nx2 matrix) - onset and offset timestamps of a N trials
%           spiketimes - a vector containing spike times
%           clusters - unit id for a given spike
%       
% OUTPUT: spikes_in_trials (Nx1 cell) - spiketimes and unit id during a
% given trial

TrialNum = size(trial_intervals,1)

spikes_in_trials = cell(TrialNum,1);

for i = 1:TrialNum
    onset = trial_intervals(i,1);
    offset = trial_intervals(i,2);
    
    idx = find(spiketimes >= onset & spiketimes < offset);
    spikes(:,1) = spiketimes(idx); % timestamps
    spikes(:,2) = clusters (idx); % Unit id
    
    spikes_in_trials{i} = spikes;
    spikes = [];
end