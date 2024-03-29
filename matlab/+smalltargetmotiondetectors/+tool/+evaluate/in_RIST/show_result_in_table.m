%show_result_in_table
if isempty(dbstack(1,'-completenames'))
    clc, clear, close all;
end

%%
%[modelList, datasetList]
get_model_and_dataset_list;

get_IO_folder;

%%
tableAUC = table(...
    'size', [length(datasetList)+1, length(modelList)], ...
    'VariableTypes', repelem({'double'}, length(modelList)), ...
    'RowNames', [datasetList, {'meanAUC(%)'}]', ...
    'VariableNames', modelList, ...
    'DimensionNames', {'dataset', 'model'} );
tableAUC.Properties.Description = ...
    'AUC(Area Under Curve) in RIST (%)';

tableMeanR = table(...
    'size', [length(datasetList)+1, length(modelList)], ...
    'VariableTypes', repelem({'double'}, length(modelList)), ...
    'RowNames', [datasetList, {'meanRPI(%)'}]', ...
    'VariableNames', modelList, ...
    'DimensionNames', {'dataset', 'model'} );
tableMeanR.Properties.Description = ...
    'RPI(Recall Per Image) in RIST (%)';

%% Waitbat
hWaitbat = waitbar(0, 'Loading data...');
%%
idxI = 0;
for datasetCell = datasetList
    hWaitbat = ...
        waitbar(idxI/length(datasetList), hWaitbat, 'Loading data...');

    idxI = idxI + 1;
    datasetName = datasetCell{1};
    %-------------------------------------%
    outputPath = [outputFolder, 'final_result/'];
    %-------------------------------------%

    idxJ = 0;
    for modelNameCell = modelList
        idxJ = idxJ + 1;
        modelName = modelNameCell{1};
        % fprintf('%s\t', modelName);
        
        

        try
            load([outputPath, datasetName, '.mat']);
            eval([modelName,'Result = RIST_result.',modelName,';']);
        catch
            continue;
        end

        meanR = eval([modelName, 'Result.meanR']);
        mTP = eval([modelName, 'Result.mTP']);
        mFP = eval([modelName, 'Result.mFP']);
        index_FP = 1;
        while mFP(index_FP) > 5 
            index_FP = index_FP + 1;
        end

        h_mTP = (mTP(index_FP:end-1)+mTP(index_FP+1:end)) / 2;
        area_mFP = mFP(index_FP:end-1) - mFP(index_FP+1:end);
        %AUC_
        AUC = dot(h_mTP, area_mFP)/(mFP(index_FP)-mFP(end));

        tableMeanR(idxI, idxJ) = {round(meanR*100, 2)};
        tableAUC(idxI, idxJ) = {round(AUC*100, 2)};

        clear RIST_result AUC meanR mTP mFP;
        eval(['clear ', modelName, 'Result']);

    end
end

%%
close(hWaitbat);

%%
tableAUC(end,:) = array2table( ...
    round( mean(table2array(tableAUC(1:end-1, :)), 1), 2)...
    );

tableMeanR(end,:) = array2table(...
    round( mean(table2array(tableMeanR(1:end-1, :)), 1), 2)...
    );

%%
areaUnderCurve = tableAUC;
recallPerImage =  tableMeanR;
%%
save('areaUnderCurve.mat', 'areaUnderCurve');
save('recallPerImage.mat', 'recallPerImage');

%%
areaUnderCurve
recallPerImage
