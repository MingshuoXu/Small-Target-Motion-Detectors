classdef STFeedbackSTMDv2 < ...
        smalltargetmotiondetectors.model.Backbonev2
    % FeedbackSTMD - Feedback Small Target Motion Detector
    %   This class implements a Feedback Extended Small Target Motion Detector
    %   by inheriting from the Backbonev2 class.
    %
    % Ref: Wang, H., Zhong, Z., Lei, F., Peng, J., & Yue, S. (2023). 
    % Bio-Inspired Small Target Motion Detection With Spatio-Temporal 
    % Feedback in Natural Scenes. IEEE Transactions on Image Processing.
    
    properties
        
    end
     

    
    methods
        function self = STFeedbackSTMDv2()
            %  Constructor method
            %   Initializes an instance of the STFeedbackSTMD class.
            %
            % Syntax:
            %   obj = FeedbackSTMD()
            %
            % Description:
            %   Initializes the FeedbackSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.Backbonev2();
            % Import necessary packages
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;
            import smalltargetmotiondetectors.core.stfeedbackstmdv2_core.*;

            % Customize Medulla and Lobula component

            self.hLobula = ...
                smalltargetmotiondetectors.core.stfeedbackstmdv2_core.Lobula();

        end

        function init(self)
            init@smalltargetmotiondetectors.model.Backbonev2(self);
            if isempty(self.inputFps)
                error('input fps is needed!');
            else
                self.hLobula.hLPTC.inputFps = self.inputFps;
            end
        end

    end
end
