classdef HaarSTMD < smalltargetmotiondetectors.model.ESTMDBackbone
    % HaarSTMD
    
    properties
        % Define properties here (if any)
    end
     
    
    methods
        function self = HaarSTMD()
            % HaarSTMD Constructor method
            %   Initializes an instance of the HaarSTMD class.
            %
            % Syntax:
            %   obj = HaarSTMD()
            %
            % Description:
            %   Initializes the HaarSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.ESTMDBackbone();
            
            % Import necessary packages
            import smalltargetmotiondetectors.core.haarstmd_core.*

            % Initialize components
            self.hMedulla = ...
                smalltargetmotiondetectors.core.haarstmd_core.Medulla();
            self.hLobula = ...
                smalltargetmotiondetectors.core.haarstmd_core.Lobula();
            
            % init parameter
            self.hLamina.hGammaBandPassFilter.hGammaDelay1.order = 10;
            self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau = 3;
            self.hLamina.hGammaBandPassFilter.hGammaDelay2.order = 10;
            self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau = 9;
            self.hLobula.hSubInhi.Sigma1 = 1.5;
            self.hLobula.hSubInhi.Sigma2 = 3;
            self.hLobula.tau = 1;
        end % [EoF]
        
        function init_config(self)           
            init_config@smalltargetmotiondetectors.model.ESTMDBackbone(self);
        end % [EoF]

        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the HaarSTMD model.
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the HaarSTMD model
            %   components (retina, lamina, medulla, and lobula) and
            %   generates the model's response.
            
            % Process input matrix through model components
            self.retinaOpt = self.hRetina.process(iptMatrix);
            self.laminaOpt = self.hLamina.process(self.retinaOpt);
            [cellSpatialOpt, temporalOpt] = self.hMedulla.process(self.laminaOpt);
            self.lobulaOpt = self.hLobula.process(cellSpatialOpt, temporalOpt);

            % Set model response
            self.modelOpt.response = self.lobulaOpt;
        end % [EoF] end of function

    end % [EoM] end of method
    
end % [EoC] end of class
