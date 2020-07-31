function [win_train, spikes] = OrganizeHMMInput (session, area, feedback)
% reformat your data into the 'spikes' structure array with dimension 
% [ntrials,nunits] and field .spk containing spike times as columns array (in seconds)
% create your own 'win_train' array with dimensions [ntrials, 2], where each row is the [start, end] times for each trial
%(spike times in 'spikes' must be consistently aligned with 'win_train')
if isempty(feedback) 
    feedback = 1:length(session.trials.intervals); % Entire trials
elseif feedback == 1
    feedback = find(session.trials.feedbackType==1);
else
    feedback = find(session.trials.feedbackType==-1);
end

included = find (session.trials.included == 1);
clean_trials = intersect(included,feedback);

spikes_in_trials = extract_spikes_in_trials(session.trials.intervals(clean_trials,:), ..., 
    session.spikes.times, session.spikes.clusters);

IDs = mappingArea(session, area);

% Organize win_train
visTime = session.trials.visualStim_times;
visTime = visTime(clean_trials);
win_train(:,1) = visTime -0.5;
win_train(:,2) = visTime +2;

% Organize spikes structure
nTrials = size(spikes_in_trials, 1);
NUnits = length(IDs);
spikes=[]; spikes(nTrials,NUnits).spk=[];


for trial = 1:nTrials
    
    thisTrialSpikes = spikes_in_trials{trial};
    for unit= 1:NUnits
        spk = thisTrialSpikes(thisTrialSpikes(:,2) ==IDs(unit));
        spikes(trial,unit).spk = spk;
    end
       
end


        
        
        
