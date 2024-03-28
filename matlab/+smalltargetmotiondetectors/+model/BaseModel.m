classdef BaseModel < handle
    % BASEMODEL - Base class for Small Target Motion Detector models
    %   This class serves as the base class for all Small Target Motion
    %   Detector models.
    %
    %   Author: Mingshuo Xu
    %   Date: 2024-01-20
    %   LastEditTime: 2024-03-06
    
    properties
        hRetina; % Handle for the retina layer
        hLamina; % Handle for the lamina layer
        hMedulla; % Handle for the medulla layer
        hLobula; % Handle for the lobula layer
        inputFps;
    end

    properties(Hidden)
        retinaOpt; % Retina layer output
        laminaOpt; % Lamina layer output
        medullaOpt; % Medulla layer output
        lobulaOpt; % Lobula layer output
        % Model output structure
        modelOpt = struct('response', [], 'direction', []); 
    end
    
    methods
        function self = BaseModel()
            % BASEMODEL Constructor
            % Get the full path of this file
            filePath = mfilename('fullpath');
            %   Find the index of '/+smalltargetmotiondetectors/'
            % in the file path
            indexPath = strfind(filePath, ...
                [filesep, '+smalltargetmotiondetectors', filesep]);
            % Add the path to the package containing the models
            addpath(filePath(1:indexPath(end)-1));

            % Import necessary packages
            import smalltargetmotiondetectors.*;
            import smalltargetmotiondetectors.core.*;
            import smalltargetmotiondetectors.tool.*;

        end

        function modelOpt = process(self, modelIpt)
            % PROCESS Method
            %   Processes the input and returns the model output.
            %
            % Syntax:
            %   modelOpt = obj.process(modelIpt)
            %
            % Input:
            %   modelIpt - Input for model processing
            %
            % Output:
            %   modelOpt - Model output structure
            
            % Call the model structure method
            self.model_structure(modelIpt);
            % Return the model output
            modelOpt = self.modelOpt;
        end

    end
    
    methods(Abstract)
        % Abstract methods to be implemented by subclasses
        init(); % Method for initializing model components
        model_structure(); % Method for defining model structure
    end
end
