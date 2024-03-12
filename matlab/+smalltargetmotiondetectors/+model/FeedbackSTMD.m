classdef FeedbackSTMD < smalltargetmotiondetectors.model.ESTMDBackbone
    % FeedbackSTMD - Feedback Small Target Motion Detector
    %   This class implements a Feedback Extended Small Target Motion Detector
    %   by inheriting from the ESTMDBackbone class.
    %
    % Ref: H. Wang, H. Wang, J. Zhao, C. Hu, J. Peng, S. Yue, A time-delay 
    % feedback neural network for discriminating small, fast-moving targets
    % in complex dynamic environments, IEEE Transactions on Neural Networks
    % and Learning Systems (2021).
    
    properties
        % Define properties here (if any)
    end
     

    
    methods
        function self = FeedbackSTMD()
            % FeedbackSTMD Constructor method
            %   Initializes an instance of the FeedbackSTMD class.
            %
            % Syntax:
            %   obj = FeedbackSTMD()
            %
            % Description:
            %   Initializes the FeedbackSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.ESTMDBackbone();
            % Import necessary packages
            import smalltargetmotiondetectors.core.feedbackstmd_core.*;

            % Customize Lobula component
            self.hLobula = ...
                smalltargetmotiondetectors.core.feedbackstmd_core.Lobula();

            % Customize Lamina's GammaBankPassFilter properties
            self.hLamina.hGammaBankPassFilter.hGammaDelay1.order = 4;
            self.hLamina.hGammaBankPassFilter.hGammaDelay1.tau = 8;
            self.hLamina.hGammaBankPassFilter.hGammaDelay2.order = 16;
            self.hLamina.hGammaBankPassFilter.hGammaDelay2.tau = 32;

            % Customize Medulla's Tm1 component properties
            self.hMedulla.hTm1.hGammaDelay.order = 9;
            self.hMedulla.hTm1.hGammaDelay.tau = 45;
        end

        
        function init(self)
            %INIT Initializes the FeedbackSTMD components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the components inherited from ESTMDBackbone.
            
            init@smalltargetmotiondetectors.model.ESTMDBackbone(self);
        end


        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the FeedbackSTMD model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the FeedbackSTMD model
            %   components (retina, lamina, medulla, and lobula) and
            %   generates the model's response.
            
            % Process input matrix through model components
            self.retinaOpt = self.hRetina.process(iptMatrix);

            self.laminaOpt = self.hLamina.process(self.retinaOpt);

            self.hMedulla.process(self.laminaOpt);
            self.medullaOpt = self.hMedulla.Opt;

            self.lobulaOpt = self.hLobula.process(self.medullaOpt);

            %%
            self.modelOpt.response = self.lobulaOpt;
        end

    end
end
