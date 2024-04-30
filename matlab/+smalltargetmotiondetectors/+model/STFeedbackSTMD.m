classdef STFeedbackSTMD < ...
        smalltargetmotiondetectors.model.ESTMDBackbone
    % FeedbackSTMD - Feedback Small Target Motion Detector
    %   This class implements a Feedback Extended Small Target Motion Detector
    %   by inheriting from the ESTMDBackbone class.
    %
    % Ref: Wang, H., Zhong, Z., Lei, F., Peng, J., & Yue, S. (2023). 
    % Bio-Inspired Small Target Motion Detection With Spatio-Temporal 
    % Feedback in Natural Scenes. IEEE Transactions on Image Processing.
    
    properties
        
    end
     

    
    methods
        function self = STFeedbackSTMD()
            %  Constructor method
            %   Initializes an instance of the STFeedbackSTMD class.
            %
            % Syntax:
            %   obj = STFeedbackSTMD()
            %
            % Description:
            %   Initializes the STFeedbackSTMD object and sets up 
            % its components.
            
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

        function init_config(self)
            init_config@smalltargetmotiondetectors.model.ESTMDBackbone(self);
        end

    end
end
