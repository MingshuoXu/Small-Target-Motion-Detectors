classdef PredictionModule < smalltargetmotiondetectors.core.BaseCore
    % PredictionModule class for ApgSTMD

    properties
        % Parameters
        velocity;           % Velocity: v_{opt} (Defalt: 0.25)
        intDeltaT = 25;     % Delta time
        sizeFilter = 25;    % Size of filter
        numFilter = 8;      % Number of filters
        zeta = 2;           % Zeta parameter
        eta = 2.5;          % Eta parameter
        kappa = 0.02;       % Kappa parameter
        mu = 0.75;          % Mu parameter
        beta = 1;           % Beta parameter
    end

    properties(Hidden)
        % Hidden properties
        predictionKernel;       % Prediction kernel
        cellPredictionGain;     % Cell array for prediction gain
        cellPredictionMap;      % Cell array for prediction map
        timeAttenuationKernel;  % kernel in formula (23)
    end

    methods
        function self = PredictionModule()
            % Constructor method
            % Initializes the PredictionModule object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes the prediction module
            
            import smalltargetmotiondetectors.tool.kernel.*;

            if self.intDeltaT < 1
                self.intDeltaT = 1;
            elseif ~isinteger(self.intDeltaT)
                self.intDeltaT = round(self.intDeltaT);
            end
            
            if isempty(self.velocity)
                self.velocity = 25 / 4 / self.intDeltaT;
            end
            
            self.predictionKernel = create_prediction_kernel(...
                self.velocity, ...
                self.intDeltaT, ...
                self.sizeFilter, ...
                self.numFilter, ...
                self.zeta, ...
                self.eta);
            self.cellPredictionGain = ...
                cell(self.intDeltaT + 1, self.numFilter);
            self.cellPredictionMap = cell(self.intDeltaT + 1, 1);

            self.timeAttenuationKernel = ...
                exp( self.kappa * ( -self.intDeltaT:0 ) );
            
        end

        function [facilitatedOpt, predictionMap] = process(self, lobulaOpt)
            % Processing method
            % Processes the input lobulaOpt to predict motion and update
            % prediction map
            import smalltargetmotiondetectors.tool.compute.*;
            
            numDict = length(lobulaOpt);
            [imgH, imgW] = size(lobulaOpt{1});

            % Prediction Gain
            self.cellPredictionGain = circshift(self.cellPredictionGain, -1);
            predictionGain = cell(numDict, 1);
            for idxD = 1:numDict
                if isempty(self.cellPredictionGain{1,idxD})
                    predictionGain{idxD} = conv2(...
                        self.mu * lobulaOpt{idxD},...
                        self.predictionKernel{idxD},...
                        'same' ...
                        );
                else
                    predictionGain{idxD} = conv2(...
                        self.mu * lobulaOpt{idxD} ...
                        + (1 - self.mu) * self.cellPredictionGain{1,idxD}, ...
                        self.predictionKernel{idxD}, ...
                        'same' ...
                        );
                end
            end

            [self.cellPredictionGain(end,:)] = deal(predictionGain);

            % Prediction Map
            tobePredictionMap = zeros(imgH, imgW);
            for idxD = 1:numDict
                tobePredictionMap = tobePredictionMap + predictionGain{idxD};
            end


            % Facilitated STMD Output
            facilitatedOpt = cell(numDict, 1);
            for idxD = 1:numDict
                facilitatedOpt{idxD} = lobulaOpt{idxD};
                if ~isempty(self.cellPredictionGain{end, idxD})
                    facilitatedOpt{idxD} = facilitatedOpt{idxD} + ...
                        self.beta * ...
                        compute_temporal_conv( ...
                        self.cellPredictionGain(:,idxD), ...
                        self.timeAttenuationKernel ...
                        );
                end

            end

            % Memorizer update
            maxTobePreMap = max(tobePredictionMap, [], 'all');
            self.cellPredictionMap = circshift(self.cellPredictionMap, -1);
            self.cellPredictionMap{end} = ...
                ( tobePredictionMap > maxTobePreMap*2e-1 );

            % Output
            predictionMap = self.cellPredictionMap{1};
            self.Opt = facilitatedOpt;
        end
    end

end


















