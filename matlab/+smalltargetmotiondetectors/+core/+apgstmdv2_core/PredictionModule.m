classdef PredictionModule < smalltargetmotiondetectors.core.BaseCore
    % PredictionModule class for ApgSTMD

    properties
        % Parameters
        velocity = 0.1;    % Velocity: v_{opt} (Defalt: 0.25)
        intDeltaT = 0;      % Delta time
        sizeFilter = 20;    % Size of filter
        numFilter = 8;      % Number of filters
        zeta = 2;           % Zeta parameter
        eta = 2.5;          % Eta parameter
        beta = 1;           % Beta parameter
    end

    properties(Hidden)
        % Hidden properties
        predictionKernel;       % Prediction kernel
    end

    methods
        function self = PredictionModule()
            % Constructor method
            % Initializes the PredictionModule object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % Initializes the prediction module
            
            import smalltargetmotiondetectors.util.kernel.*;
            
            self.predictionKernel = create_prediction_kernel(...
                self.velocity, ...
                self.intDeltaT, ...
                self.sizeFilter, ...
                self.numFilter, ...
                self.zeta, ...
                self.eta...
                );
            
        end

        function [facilitatedOpt, predictionMap] = process(self, lobulaOpt)
            % Processing method
            % Processes the input lobulaOpt to predict motion and update
            % prediction map
            
            numDict = length(lobulaOpt);
            [imgH, imgW] = size(lobulaOpt{1});

            predictionGain = cell(numDict, 1);
            for idxD = 1:numDict
                predictionGain{idxD} = conv2(...
                    lobulaOpt{idxD},...
                    self.predictionKernel{idxD}, ...
                    'same' ...
                    );
            end

            % Prediction Map
            predictionMap = zeros(imgH, imgW);
            for idxD = 1:numDict
                predictionMap = predictionMap + predictionGain{idxD};
            end

            % Facilitated STMD Output
            facilitatedOpt = cell(numDict, 1);
            for idxD = 1:numDict
                facilitatedOpt{idxD} = ...
                    lobulaOpt{idxD} + self.beta*predictionGain{idxD};
            end

            % Memorizer update
            maxPreMap = max(predictionMap, [], 'all');
            predictionMap = ( predictionMap > maxPreMap*2e-1 );

            % Output
            self.Opt = facilitatedOpt;
        end
    end

end


















