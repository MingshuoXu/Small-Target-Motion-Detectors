%% show_average_running_time
clc, clear, close all;

%%
modelList = {...
    'ESTMD', ...
    'DSTMD', ...
    'STMDPlus', ...
    'ApgSTMD', ...
    'FracSTMD', ...
    'STMDv2', ...
    'FSTMD' ,...
    'FeedbackSTMD' ...
    };

datasetList = {...
    'GX010071-1', 'GX010220-1', 'GX010228-1', 'GX010230-1', ...
    'GX010231-1', 'GX010241-1', 'GX010250-1', 'GX010266-1', ...
    'GX010290-1', 'GX010291-1', 'GX010303-1', 'GX010307-1', ...
    'GX010315-1', 'GX010321-1', 'GX010322-1', 'GX010327-1', ...
    'GX010335-1', 'GX010336-1', 'GX010337-1' };

get_IO_folder;

%%
hWaitbat = waitbar(0, 'Loading data...');

%%
tableAverageRunningTime = table(...
    'size', [length(datasetList)+1, length(modelList)], ...
    'VariableTypes', repelem({'double'}, length(modelList)), ...
    'RowNames', [datasetList, {'meanART (ms/image)'}]', ...
    'VariableNames', modelList, ...
    'DimensionNames', {'dataset', 'model'} );
tableAverageRunningTime.Properties.Description = ...
    'ART(Average Running Time) in RIST (ms/image)';
%ART(AverageRunningTime)

idxI = 0;
for datasetCell = datasetList
    hWaitbat = ...
        waitbar(idxI/length(datasetList), hWaitbat, 'Loading data...');
    idxI = idxI + 1;
    datasetName = datasetCell{1};
    %-------------------------------------%
    outputPath = [outputFolder, datasetName, '/'];
    %-------------------------------------%
    
    idxJ = 0;
    for modelNameCell = modelList
        idxJ = idxJ + 1;
        modelName = modelNameCell{1};
        % fprintf('%s\t', modelName);
        
        load([outputPath, modelName, '_result.mat'], 'averageRunningTime');

        tableAverageRunningTime(idxI, idxJ) = {round(averageRunningTime*1000, 1)};

        eval(['clear ', modelName, 'Output averageRunningTime totalRunningTime;'])
    end

end

close(hWaitbat);

tableAverageRunningTime(end,:) = array2table(...
    round( mean(table2array(tableAverageRunningTime(1:end-1, :)), 1), 1)...
    );

%%
averageRunningTime = tableAverageRunningTime;

%%
save('averageRunningTime.mat', 'averageRunningTime');

%%
averageRunningTime















