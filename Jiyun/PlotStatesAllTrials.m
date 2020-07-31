function [hmm_results, hmm_postfit] = PlotStatesAllTrials(session, trial_idx, hmm_bestfit, HmmParam, total_spikes, win_train, colors)

%% HMM decoding
hmm_results=hmm.fun_HMM_decoding(total_spikes(trial_idx,:),hmm_bestfit,HmmParam,win_train(trial_idx,:));
% HMM ADMISSIBLE STATES -> state sequences
hmm_postfit=hmm.fun_HMM_postfit(total_spikes(trial_idx,:),hmm_results,HmmParam,win_train(trial_idx,:));


%%
included = find(session.trials.included == 1);
visStim = session.trials.visualStim_times;
visStim = visStim(included);
response = session.trials.response_times;
response = response(included);
feedbacktime = session.trials.feedback_times;
feedbacktime = feedbacktime(included);
goCue = session.trials.goCue_times;
goCue = goCue(included);
%%

numTrials = length(trial_idx);
figure
for i = 1:numTrials
    this_sequence = hmm_postfit(i).sequence;
    numTrans = size(this_sequence,2);
    interval = win_train(trial_idx(i),:);
    init_factor = interval(1)+0.5; % correction factor to set time to -0.5 to 2 s, time 0 is cue
    
   
    
    responseTime = response(trial_idx(i))- init_factor;
    feedbackTime = feedbacktime(trial_idx(i))- init_factor;
    cuetime = goCue(trial_idx(i))- init_factor;
    
    for j= 1:numTrans
        on = this_sequence(1, j) - init_factor;
        off = this_sequence(2, j) - init_factor;
        state = this_sequence(4, j)
        plot([on off], [i, i], 'LineWidth', 3, 'Color', colors(state,:))
        hold on
    end
    
    %plot(cuetime, i, 'bo', 'Markersize', 5)
    plot(responseTime, i, 'ro', 'Markersize', 5, 'MarkerFaceColor', 'r') 
    %plot(feedbackTime, i, 'gs', 'Markersize', 5)
end



ylabel('Trial #')
xlabel('Time from Cue onset (s)')
yvals = get(gca,'ylim')
plot([0 0], yvals, 'k-', 'linewidth',3)
xlim([-0.5 2])
ylim([0 length(trial_idx)])
text(0.1, yvals(2)-10, 'visual stim', 'color', 'k')
text(1, yvals(2)-10, 'response', 'color', 'r')