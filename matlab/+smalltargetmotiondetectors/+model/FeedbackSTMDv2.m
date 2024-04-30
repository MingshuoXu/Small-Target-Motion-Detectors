classdef FeedbackSTMDv2 < smalltargetmotiondetectors.model.Backbonev2
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
        function self = FeedbackSTMDv2()
            % FeedbackSTMD Constructor method
            %   Initializes an instance of the FeedbackSTMD class.
            %
            % Syntax:
            %   obj = FeedbackSTMD()
            %
            % Description:
            %   Initializes the FeedbackSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.Backbonev2();

            import smalltargetmotiondetectors.core.feedbackstmdv2_core.*;
            self.hLobula = ...
                smalltargetmotiondetectors.core.feedbackstmdv2_core.Lobula();
        end

        
        function init_config(self)
            %INIT Initializes the FeedbackSTMD components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the components inherited from ESTMDBackbone.
            
            init_config@smalltargetmotiondetectors.model.Backbonev2(self);
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

            [self.lobulaOpt, self.modelOpt.direction] = ...
                self.hLobula.process(...
                self.medullaOpt{1}, self.medullaOpt{2});

            %%
            self.modelOpt.response = self.lobulaOpt;
        end

        
    end
end
