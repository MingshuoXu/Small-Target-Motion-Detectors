%compute_model_output
if isempty(dbstack(1,'-completenames'))
    clc, clear, close all;
end

%%
% Get the full path of this file
filePath = mfilename('fullpath');
%   Find the index of '/matlab/+smalltargetmotiondetectors/'
% in the file path
indexPath = strfind(filePath, ...
    [filesep, 'matlab', filesep, '+smalltargetmotiondetectors', filesep]);
% Add the path to the package containing the models
addpath(filePath(1:indexPath(end)+7));

%%
import smalltargetmotiondetectors.*;
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.tool.*;
import smalltargetmotiondetectors.model.*;


%% original
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
    'ESTMD', ...
    'DSTMD', ...
    'FracSTMD', ...
    'STMDPlus', ...
    'ApgSTMD', ...
    'STMDv2', ...
    'FeedbackSTMD', ...
    'FSTMD' ...
    };

datasetList = {'GX010321-1', 'GX010322-1', 'GX010327-1', ...
    'GX010335-1', 'GX010336-1', 'GX010337-1' };

%%
get_IO_folder;

hNMS = smalltargetmotiondetectors.tool.MatrixNMS(5, 'auto');

%%
for datasetCell = datasetList
    datasetName = datasetCell{1};
    fprintf('\n%s\t', datasetName);

    %-------------------------------------%
    datasetFormat = ...
        [folderRIST,'/',datasetName,'/',datasetName,'.mp4'];

    outputPath = [outputFolder, datasetName, '/'];
    %-------------------------------------%

    if ~exist(outputPath, 'dir')
        mkdir(outputPath);
    end

    speedCoeff = get_speedcoef(datasetName);

    for modelNameCell = modelList
        modelName = modelNameCell{1};
        fprintf('%s ', modelName);
        hModel = instancing_model(modelName);
        %% parameters
        hModel = adjust_coef(hModel, modelName, speedCoeff);

        %%
        hModel.init();

        hImgSteam = VidstreamReader(datasetFormat);
        hImgSteam.isShowWaitbar = true;


        eval([modelName, 'Output = cell(hImgSteam.endFrame, 1);']);

        totalRunningTime = 0;
        for frameIdx = 1:hImgSteam.endFrame
            [grayImg, ~] = hImgSteam.get_next_frame();
            tic0 = tic;
            % inference-----%
            result = hModel.process(grayImg);
            %---------------%
            totalRunningTime = totalRunningTime + toc(tic0);

            response = result.response;
            maxOpt = max(response(:));
            if maxOpt > 0
                normOutput = response ./ maxOpt;
                nmsMatrix = hNMS.nms(normOutput);

                sparseOutput = sparse(nmsMatrix);
            else
                [m, n] = size(response);
                sparseOutput = sparse(m, n);
            end

            eval([modelName, 'Output{frameIdx} = sparseOutput;']);

        end
        averageRunningTime = totalRunningTime / hImgSteam.endFrame;

        fprintf('%.1f ms\t', averageRunningTime*1000)

        save([outputPath, modelName, '_result.mat'],...
            [modelName, 'Output'], 'averageRunningTime', ...
            '-nocompression', '-v7.3');


        clear hModel hImgSteam;
        eval(['clear ', modelName, 'Output;'])
    end
end














