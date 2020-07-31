load('HMM_MOs_Total_trials.mat')
%% HMM trained on all trials (n=201)

colors=aux.distinguishable_colors(max(HmmParam.VarStates,4));
% plot tpm and epm
hmm.plot_tpm_epm;


%%
filepath = 'C:\Users\Larkum_Practical_02\Documents\GitHub\NMA2020_group_project\data\Cori_2016-12-14';
Cori1 = loadSession(filepath);


included = find (Cori1.trials.included == 1);
feedbackType = Cori1.trials.feedbackType;
feedbackType = feedbackType(included);

Hits = find(feedbackType==1);
Misses = find(feedbackType==-1);


%% Hit trials

PlotStatesAllTrials(Cori1, Hits, hmm_bestfit, HmmParam, total_spikes, win_train, colors)

title('Hit trials')
%% Miss trials
PlotStatesAllTrials(Cori1, Misses, hmm_bestfit, HmmParam, total_spikes, win_train, colors)

title('Miss trials')

%%
RightContrast = Cori1.trials.visualStim_contrastRight;
RightContrast = RightContrast(included);
LeftContrast = Cori1.trials.visualStim_contrastLeft;
LeftContrast = LeftContrast(included);

RightHigh = find(RightContrast > LeftContrast);
LeftHigh = find(RightContrast < LeftContrast);

RightHighHits = intersect(RightHigh,Hits);
LeftHighHits = intersect(LeftHigh,Hits);

[hmm_results_right, hmm_postfit_right] = PlotStatesAllTrials(Cori1, RightHighHits, hmm_bestfit, HmmParam, total_spikes, win_train, col)
title('Vis = Right, Hits')

[hmm_results_left, hmm_postfit_left] = PlotStatesAllTrials(Cori1, LeftHighHits, hmm_bestfit, HmmParam, total_spikes, win_train, col)
title('Vis = Left, Hits')

%%
PlotParam=[];
for itrial=5:10
    %figure(3); clf;
    figure
    %filename=fullfile(hmmdir,['Plot_trial' num2str(itrial) '.pdf']);
    PlotParam=struct('win',win_train(RightHighHits(itrial),:),'colors',col,'fntsz',8,'gnunits',gnunits);
    DATA=struct('win',win_train(RightHighHits(itrial),:),'seq',hmm_postfit_right(itrial).sequence,'pstates',...
        hmm_results_right(itrial).pStates,'rates',hmm_results_right(itrial).rates);
    DATA.Spikes(1:gnunits)=total_spikes(Hits(itrial),1:gnunits);
%     figure(1); clf;
    hmm.fun_HMMRasterplot(DATA,HmmParam,PlotParam);
    %saveas(gcf,filename,'pdf');
end