classdef FeedbackPathway < smalltargetmotiondetectors.core.BaseCore
    % FeedbackPathway class for the feedback pathway

    
    properties
        hGammaDelay;
        feedbackConstant = 0.22; % Feedback constant a (formula (4))
    end

    methods
        function self = FeedbackPathway()
            % Constructor method
            % Initializes the FeedbackPathway object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.GammaDelay;
            
            self.hGammaDelay = GammaDelay(5, 10);
        end
    end
    
    methods
        function init(self)
            % Initialization method
            % Initializes the GammaDelay object
            
            self.hGammaDelay.init();
        end
        
        function feedbackOpt = process(self, feedbackIpt)
            % Processing method
            % Processes the feedbackIpt to generate the feedback output
            
            feedbackOpt = ...
                self.feedbackConstant * ...
                self.hGammaDelay.process(feedbackIpt);
            self.Opt = feedbackOpt;
        end
    end
end
