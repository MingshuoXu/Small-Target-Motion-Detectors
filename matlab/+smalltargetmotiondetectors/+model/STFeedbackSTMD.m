classdef STFeedbackSTMD < ...
        smalltargetmotiondetectors.model.ESTMDBackbone
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
        function self = STFeedbackSTMD()
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
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;

            % Customize Medulla and Lobula component
            self.hMedulla = ...
                smalltargetmotiondetectors.core.stfeedbackstmd_core.Medulla();
            self.hLobula = ...
                smalltargetmotiondetectors.core.stfeedbackstmd_core.Lobula();

        end

        function init(self)
            init@smalltargetmotiondetectors.model.ESTMDBackbone(self);
            if isempty(self.inputFps)
                error('input fps is needed!');
            else
                self.hLobula.hLPTC.inputFps = self.inputFps;
            end
        end

    end
end
