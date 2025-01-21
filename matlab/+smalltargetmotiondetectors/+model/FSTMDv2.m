classdef FSTMDv2 < smalltargetmotiondetectors.model.STMDNet
    % FSTMD - Feedback Small Target Motion Detector
    %   This class implements a Feedback Small Target Motion Detector by
    %   inheriting from the ESTMDBackbone class.
    %
    % Ref: Ling J, Wang H, Xu M, Chen H, Li H and Peng J (2022) 
    % Mathematical study of neural feedback roles in small target motion 
    % detection. Front. Neurorobot. 16:984430. 
    % doi: 10.3389/fnbot.2022.984430
    
    properties
        hFeedbackPathway; % Handle for the feedback pathway
        iterationThres = 1e-3; % Iteration threshold for feedback loop
        maxIteraNum = 100; % Maximum number of iterations for feedback loop
    end
    properties(Hidden)
        feedbackSignal; % Hidden property for storing feedback signal
    end    

    
    methods
        function self = FSTMDv2()
            % FSTMD Constructor method
            %   Initializes an instance of the FSTMD class.
            %
            % Syntax:
            %   obj = FSTMD()
            %
            % Description:
            %   Initializes the FSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.STMDNet();

            % Import necessary packages
            import smalltargetmotiondetectors.core.fstmd_core.*;
            import smalltargetmotiondetectors.core.fstmdv2_core.*

            self.hLamina = ...
                smalltargetmotiondetectors.core.fstmdv2_core.Lamina();
            % Initialize feedback pathway component
            self.hFeedbackPathway = ...
                smalltargetmotiondetectors.core.fstmd_core.FeedbackPathway();

            self.hFeedbackPathway.hGammaDelay.tau = 1;
        end
        
        

        function init_config(self)
            % INIT Method
            %   Initializes the FSTMD components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the components inherited from ESTMDBackbone
            %   as well as the feedback pathway.
            
            % Call superclass init method
            init_config@smalltargetmotiondetectors.model.STMDNet(self);
            % Initialize feedback pathway
            self.hFeedbackPathway.init_config();
        end

        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the FSTMD model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the FSTMD model
            %   components (retina, lamina, medulla, lobula, 
            % and feedback pathway)
            %   and generates the model's response.
            
            [m, n] = size(iptMatrix);
            lastFeedback = ones(m, n);
            self.feedbackSignal = zeros(m, n);

            %% Retina layer
            self.retinaOpt = self.hRetina.process(iptMatrix);

            %% Feedback loop
            iterationCount = 1;
            self.set_loop_state(false);
            while iterationCount < self.maxIteraNum ...
                && max(abs(self.feedbackSignal-lastFeedback), [], 'all') ...
                > self.iterationThres
                    
                lastFeedback = self.feedbackSignal;
                
                self.laminaOpt = ...
                    self.hLamina.process(self.retinaOpt + self.feedbackSignal);
                self.hMedulla.process(self.laminaOpt);
                self.medullaOpt = self.hMedulla.Opt;

                [self.lobulaOpt, direction, correlationOpt]...
                    = self.hLobula.process(...
                    self.medullaOpt{1},...
                    self.medullaOpt{2},...
                    self.laminaOpt);
                self.feedbackSignal = ...
                    self.hFeedbackPathway.process(correlationOpt);

                iterationCount = iterationCount + 1;
                self.set_loop_state(true);
            end
            
            %% Set model response
            self.modelOpt.response = self.lobulaOpt;
            self.modelOpt.direction = direction;
        end

        function set_loop_state(self, state)
            % Disable circshift for certain components
            self.hLamina.isInLoop = state; 
            self.hFeedbackPathway.hGammaDelay.isInLoop = state;
        end

    end

end
