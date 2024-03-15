function model = instancing_model(modelName, modelPara)
    % INSTANCING_MODEL - Instantiate a model object based on the given model name.
    %
    % Syntax:
    %   model = instancing_model(modelName, modelPara)
    %
    % Input:
    %   modelName - Name of the model to instantiate.
    %   modelPara - Parameters for model instantiation (optional).
    %
    % Output:
    %   model - The instantiated model object.
    %
    % Explanation:
    %   This function creates an instance of a model object based on the 
    % specified model name. It supports loading the model from a file or 
    % providing the model name directly.
    %
    % Example:
    %   model = instancing_model('STMDv2');
    
    % Get the full path of this file
    filePath = mfilename('fullpath');
    %   Find the index of '/matlab/+smalltargetmotiondetectors/'
    % in the file path
    indexPath = strfind(filePath, ...
        [filesep, 'matlab', filesep, '+smalltargetmotiondetectors', filesep]);
    % Add the path to the package containing the models
    addpath(filePath(1:indexPath(end)+7));

    % Import the necessary packages
    import smalltargetmotiondetectors.model.*;

    % Determine if GUI should be opened for model selection
    if ~nargin || isempty(modelName)
        isOpenUI = true;
    else
        isOpenUI = false;
    end

    % Open GUI for model selection if necessary
    if isOpenUI
        % Open file dialog for model selection
        [modelName] = uigetfile(...
            {'*.*'}, 'Pick a model from M-file or P-file', ...
            [filePath(1:indexPath+35), '+model\STMDv2.p'] );

        % Check if BaseModel is selected (it's an abstract class)
        if strcmp(modelName, 'BaseModel.m')
            error(['BaseModel is an Abstract Class! ',...
                'Please select another model.']);
        end

        % Check if a model is selected
        if ~modelName
            error('Please re-run and select a model.');
        end
        % Remove file extension from the model name
        modelName(end-1:end) = [];
    end

    % Instantiate the model
    model = eval([modelName, '();']);

    % Process additional parameters if provided
    if nargin == 2
        % Handle model parameters
        % modelPara;
    end
end
