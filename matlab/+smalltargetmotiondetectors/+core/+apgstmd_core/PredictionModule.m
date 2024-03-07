classdef PredictionModule < smalltargetmotiondetectors.core.BaseCore
    % PredictionModule class for ApgSTMD

    properties
        % Parameters
        vel = 0.25;         % Velocity
        deltaT = 25;        % Delta time
        sizeFilter = 25;    % Size of filter
        numFilter = 8;      % Number of filters
        zeta = 2;           % Zeta parameter
        eta = 2.5;          % Eta parameter
        kappa = 0.02;       % Kappa parameter
        mu = 0.25;          % Mu parameter
        beta = 1;           % Beta parameter
    end

    properties(Hidden)
        % Hidden properties
        predictionKernel;   % Prediction kernel
        cellPredictionGain; % Cell array for prediction gain
        cellPredictionMap;  % Cell array for prediction map
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
            self.predictionKernel = create_prediction_kernel(...
                self.vel, ...
                self.deltaT, ...
                self.sizeFilter, ...
                self.numFilter, ...
                self.zeta, ...
                self.eta);
            self.cellPredictionGain = cell(self.deltaT + 1, 1);
            self.cellPredictionMap = cell(self.deltaT + 1, 1);
        end

        function [facilitatedOpt, predictionMap] = process(self, lobulaOpt)
            % Processing method
            % Processes the input lobulaOpt to predict motion and update
            % prediction map
            
            numDict = length(lobulaOpt);
            [imgH, imgW] = size(lobulaOpt{1});

            % Prediction Gain
            predictionGain = cell(numDict, 1);
            self.cellPredictionGain(1) = [];
            for idx = 1:numDict
                if isempty(self.cellPredictionGain{1})
                    predictionGain{idx} = conv2(...
                        self.mu * lobulaOpt{idx},...
                        self.predictionKernel{idx},...
                        'same');
                else
                    predictionGain{idx} = conv2(...
                        (self.mu) * lobulaOpt{idx} + ...
                        (1 - self.mu) * self.cellPredictionGain{1}{idx}, ...
                        self.predictionKernel{idx}, ...
                        'same');
                end
            end
            self.cellPredictionGain{end + 1} = predictionGain;

            % Prediction Map
            tobePredictionMap = zeros(imgH, imgW);
            for idx = 1:numDict
                tobePredictionMap = tobePredictionMap + predictionGain{idx};
            end
            maxTobe = max(tobePredictionMap, [], 'all');
            tobePredictionMap(tobePredictionMap < maxTobe / 5) = 0;

            % Facilitated STMD Output
            facilitatedOpt = cell(numDict, 1);
            for idx = 1:numDict
                facilitatedOpt{idx} = lobulaOpt{idx};
                for ss = 0:self.deltaT
                    ids = ss + 1;
                    if ~isempty(self.cellPredictionGain{ids})
                        facilitatedOpt{idx} = facilitatedOpt{idx} ...
                            + self.beta * exp(self.kappa * (1 - ids)) *...
                            self.cellPredictionGain{ids}{idx};
                    end
                end
            end

            % Memorizer update
            self.cellPredictionMap(1) = [];
            self.cellPredictionMap{end + 1} = logical(tobePredictionMap);

            % Output
            predictionMap = self.cellPredictionMap{1};
            self.Opt = facilitatedOpt;
        end
    end

end
