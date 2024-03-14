%compute_ROC_metrics
if isempty(dbstack(1,'-completenames'))
    clc, clear, close all;
end

%% 
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of 'Small-Target-Motion-Detectors'
% in the file path
indexPath = strfind(filePath, ...
    '/matlab/+smalltargetmotiondetectors/');
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));

%%
import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.tool.*;
import smalltargetmotiondetectors.tool.evaluate.*;
import smalltargetmotiondetectors.model.*;

% %%
% modelList = {...
%     'ESTMD', ...
%     'DSTMD', ...
%     'FracSTMD', ...
%     'STMDPlus', ...
%     'ApgSTMD', ...
%     'STMDv2', ...
%     'FeedbackSTMD', ...
%     'FSTMD' ...
%     };
% 
% datasetList = {...
%     'GX010071-1', 'GX010220-1', 'GX010228-1', 'GX010230-1', ...
%     'GX010231-1', 'GX010241-1', 'GX010250-1', 'GX010266-1', ...
%     'GX010290-1', 'GX010291-1', 'GX010303-1', 'GX010307-1', ...
%     'GX010315-1', 'GX010321-1', 'GX010322-1', 'GX010327-1', ...
%     'GX010335-1', 'GX010336-1', 'GX010337-1' };
%%
modelList = {...
    'ApgSTMD' ...
    };

datasetList = {...
    'GX010071-1', 'GX010220-1', 'GX010228-1', 'GX010230-1', ...
    'GX010231-1', 'GX010241-1', 'GX010250-1', 'GX010266-1', ...
    'GX010290-1', 'GX010291-1', 'GX010303-1', 'GX010307-1', ...
    'GX010315-1', 'GX010321-1', 'GX010322-1', 'GX010327-1', ...
    'GX010335-1', 'GX010336-1', 'GX010337-1' };
%%
get_IO_folder;

saveResultPath = [outputFolder,'final_result/'];


%% STMD_model_in_RIST

for datasetCell = datasetList
    datasetName = datasetCell{1};
    fprintf('\n%s\t', datasetName);

    frameRange = get_RIST_frame_range(datasetName);

    for modelNameCell = modelList
        modelName = modelNameCell{1};
        fprintf('%s\t', modelName);
        %-------------------------------------%
        load([outputFolder, datasetName, '/', modelName, '_result.mat']);

        %-------------------------------------%
        annoFileList = dir([folderRIST, datasetName, '/*annotation.mat']);
        load([folderRIST, datasetName, '/', annoFileList(1).name]);
        bboxData = annotation.bbox;
        %-------------------------------------%
        %%
        

        output = eval([modelName, 'Output']);

        lenOutput = length(output);

        %%
        thresholdNum = 20;

        [mTP,mFN,mFP] = deal(zeros(thresholdNum,1));
        [listTP,listFN,listFP] = deal(cell(thresholdNum,1));
        
        for kk = 1:thresholdNum
            threshlod = (kk-1)/thresholdNum;
            

            logicOutput = cell(lenOutput,1);

            for tt = 1:lenOutput
                Opt = output{tt};
                Opt(Opt<threshlod) = 0;
                logicOutput{tt} =  logical(Opt);
            end


            [listTP{kk},listFN{kk},listFP{kk}] =...
                evaluation_output(logicOutput, bboxData, 'STMD');
            mTP(kk) = mean(listTP{kk}(frameRange));
            mFN(kk) = mean(listFN{kk}(frameRange));
            mFP(kk) = mean(listFP{kk}(frameRange));

        end

        thresholdList = (1:thresholdNum-1)/thresholdNum; 
        idx = (thresholdList >= 0.5);
        P = mTP(idx) ./ (mTP(idx)+mFP(idx));
        R = mTP(idx) ./ (mTP(idx)+mFN(idx));
        P(isnan(P)) = 0;
        meanP = mean(P);
        meanR = mean(R);

        save_name = [saveResultPath, datasetName, '.mat'];
        if exist(save_name, 'file')
            load(save_name);
        else
            fid = fopen(save_name, 'w');
            fclose(fid);
        end

        eval([modelName,'.mTP = mTP;'])
        eval([modelName,'.mFN = mFN;'])
        eval([modelName,'.mFP = mFP;'])
        eval([modelName,'.P = P;'])
        eval([modelName,'.R = R;'])
        eval([modelName,'.meanP = meanP;'])
        eval([modelName,'.meanR = meanR;'])
        eval([modelName,'.listTP = listTP;'])
        eval([modelName,'.listFN = listFN;'])
        eval([modelName,'.listFP = listFP;'])

        RIST_result.(modelName) = eval([modelName,';']);

        save(save_name,'RIST_result');

        eval(['clear ', modelName, '_output Output RIST_result']);
        eval(['clear ', modelName]);
    end
end
