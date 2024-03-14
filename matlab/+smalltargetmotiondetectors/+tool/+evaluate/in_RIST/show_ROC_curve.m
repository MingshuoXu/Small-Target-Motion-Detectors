%show_ROC_curve (FPPI_R_curve)
% This function plots the Receiver Operating Characteristic (ROC) curves 
%   for different models. It loads the result data for each dataset and plots
%   the ROC curve for each model in the list. 
%
% The x-axis represents the False Positive Per Image (FPPI) and the 
%   y-axis represents the True Positive Rate (TPR / Recall).
%
% Each dataset is represented by a separate figure, 
%   and each model's ROC curve is plotted on the same figure.

clc, clear, close all

%%

modelList = {...
    'ESTMD', ...
    'DSTMD', ...
    'FracSTMD', ...
    'STMDPlus', ...
    'ApgSTMD', ...
    'STMDv2', ...
    'FeedbackSTMD', ...
    'FSTMD' ...
    };

datasetList = {...
        'GX010071-1', 'GX010220-1', 'GX010228-1', 'GX010230-1', ...
        'GX010231-1', 'GX010241-1', 'GX010250-1', 'GX010266-1', ...
        'GX010290-1', 'GX010291-1', 'GX010303-1', 'GX010307-1', ...
        'GX010315-1', 'GX010321-1', 'GX010322-1', 'GX010327-1', ...
        'GX010335-1', 'GX010336-1', 'GX010337-1' };

figIdx = 170;

resultPath = 'E:/RIST_result/final_result/';

%%
for datasetStr = datasetList

    dataset = datasetStr{1};
    load([resultPath, dataset,'.mat']);

    figIdx = figIdx + 1;
    figure(figIdx);

    %%
    for modelNameCell = modelList
        modelName = modelNameCell{1};
        eval(['recall', modelName, ' = ', ... 
            'RIST_result.', modelName, '.mTP ./ ', ...
            '(RIST_result.', modelName, '.mTP + ', ...
            'RIST_result.', modelName,'.mFN);']);
        eval(['fppi_', modelName ,' = RIST_result.', modelName ,'.mFP;']);
        eval(['plot(fppi_', modelName ,', recall', modelName ,');']);
        hold on;
    end
    
    title(sprintf('ROC in %s', dataset));
    xlim([0, 30]);
    ylim([0, 1]);
    legend(modelList);
end


