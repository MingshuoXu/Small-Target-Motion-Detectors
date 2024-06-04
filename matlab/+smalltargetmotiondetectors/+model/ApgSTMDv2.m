classdef ApgSTMDv2 < smalltargetmotiondetectors.model.STMDPlusv2
    %ApgSTMD - Attention-Prediction-based Small Target Motion Detector
    %   This class implements an Attention-Prediction-based Small Target
    %   Motion Detector by inheriting from the STMDPlus class.
    %
    % Ref: H. Wang, J. Zhao, H. Wang, C. Hu, J. Peng, S. Yue, Attention
    % and predictionguided motion detection for low-contrast small moving
    % targets, IEEE Transactions on Cybernetics (2022).
    
    properties
        hAttentionPathway; % Handle to the attention pathway module
        hPredictionPathway; % Handle to the prediction pathway module
    end
    
    properties(Hidden)
        attentionOpt; % Attention pathway output
        predictionOpt; % Prediction pathway output
        predictionMap; % Prediction map
    end
    
    methods
        function self = ApgSTMDv2()
            % ApgSTMD Constructor method
            %   Initializes an instance of the ApgSTMD class.
            %
            % Syntax:
            %   obj = ApgSTMD()
            %
            % Description:
            %   Initializes the ApgSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.STMDPlusv2();
            % Import necessary packages
            import smalltargetmotiondetectors.core.apgstmd_core.*;
            import smalltargetmotiondetectors.core.apgstmdv2_core.*;
            
            % Initialize attention pathway and prediction pathway components
            self.hAttentionPathway = ...
                smalltargetmotiondetectors.core.apgstmd_core.AttentionModule();
            self.hPredictionPathway = ...
                smalltargetmotiondetectors.core.apgstmdv2_core.PredictionModule();
            
            % Set properties of Lobula's LateralInhibition module
            self.hLobula.hSubInhi.B = 3.5;
            self.hLobula.hSubInhi.Sigma1 = 1.25;
            self.hLobula.hSubInhi.Sigma2 = 2.5;
            self.hLobula.hSubInhi.e = 1.2;
            
        end
        
        function init_config(self)
            % INIT Method
            %   Initializes the ApgSTMD components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the attention pathway and prediction pathway components.
            
            % Call superclass init method
            init_config@smalltargetmotiondetectors.model.STMDPlusv2(self);
            
            % Initialize attention pathway and prediction pathway components
            self.hAttentionPathway.init_config();
            self.hPredictionPathway.init_config();
            
        end
        
        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the ApgSTMD model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the ApgSTMDv2 model
            %   components and generates the model's response.
            
            % Import compute function
            import smalltargetmotiondetectors.util.compute.*;
            import smalltargetmotiondetectors.core.backbonev2_core.*;
            
            % Preprocessing Module
            self.retinaOpt = self.hRetina.process(iptMatrix);
            
            % Attention Module
            self.attentionOpt = self.hAttentionPathway.process( ...
                self.retinaOpt, ...
                self.predictionMap);
            
            % STMD-based Neural Network
            self.laminaOpt = self.hLamina.process(self.attentionOpt);
            self.hMedulla.process(self.laminaOpt);
            self.medullaOpt = self.hMedulla.Opt;
            
            [self.lobulaOpt, self.modelOpt.direction]...
                = self.hLobula.process(...
                self.medullaOpt{1},...
                self.medullaOpt{2},...
                self.laminaOpt);
            
            % STMDPlus
            self.direContrastOpt = ...
                self.hContrastPathway.process(self.retinaOpt);
            self.mushroomBodyOpt = self.hMushroomBody.process( ...
                self.lobulaOpt, ...
                self.direContrastOpt);
            
            % Prediction Module
            multiDirectoinOpt = self.get_multi_direction_opt(...
                self.mushroomBodyOpt, ...
                self.hLobula.hLPTC.lptcMatrix);
            % self.predictionOpt is the facilitated STMD
            %   output Q(x; y; t; theta) in formula (23)
            [self.predictionOpt, self.predictionMap] = ...
                self.hPredictionPathway.process(multiDirectoinOpt);
            
            % Compute response and direction
            self.modelOpt.response = compute_response(self.predictionOpt);
        end
    end
    
    
    methods(Static)
        function multiDireOpt = get_multi_direction_opt(mushroomBodyOpt, direMatrix)
            multiDireOpt = cell(1, 8);
            [m, n] = size(mushroomBodyOpt);
            
            % Right
            direCoeff = zeros(m, n);
            direCoeff(:, 1:end-1) = direMatrix(:, 2:end);
            multiDireOpt{1} = mushroomBodyOpt .* direCoeff;
            
            % UpRight
            direCoeff = zeros(m, n);
            direCoeff(2:end, 1:end-1) = direMatrix(1:end-1, 2:end);
            multiDireOpt{2} = mushroomBodyOpt .* direCoeff;
            
            % Up
            direCoeff = zeros(m, n);
            direCoeff(2:end, :) = direMatrix(1:end-1, :);
            multiDireOpt{3} = mushroomBodyOpt .* direCoeff;
            
            % UpLeft
            direCoeff = zeros(m, n);
            direCoeff(2:end, 2:end) = direMatrix(1:end-1, 1:end-1);
            multiDireOpt{4} = mushroomBodyOpt .* direCoeff;
            
            % Left
            direCoeff = zeros(m, n);
            direCoeff(:, 2:end) = direMatrix(:, 1:end-1);
            multiDireOpt{5} = mushroomBodyOpt .* direCoeff;
            
            % BottomLeft
            direCoeff = zeros(m, n);
            direCoeff(1:end-1, 2:end) = direMatrix(2:end, 1:end-1);
            multiDireOpt{6} = mushroomBodyOpt .* direCoeff;
            
            % Bottom
            direCoeff = zeros(m, n);
            direCoeff(1:end-1, :) = direMatrix(2:end, :);
            multiDireOpt{7} = mushroomBodyOpt .* direCoeff;
            
            % BottomRight
            direCoeff = zeros(m, n);
            direCoeff(1:end-1, 1:end-1) = direMatrix(2:end, 2:end);
            multiDireOpt{8} = mushroomBodyOpt .* direCoeff;
        end
    end
    
    
end









