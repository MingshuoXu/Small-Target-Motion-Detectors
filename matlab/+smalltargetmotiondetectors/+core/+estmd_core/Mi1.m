classdef Mi1 < smalltargetmotiondetectors.core.BaseCore
    % MI1 
    
    properties
        hGammaDelay;  % Handle to the GammaDelay object
    end

    methods
        function self = Mi1()
            % Constructor method
            % Initializes the Mi1 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.GammaDelay;

            self.hGammaDelay = GammaDelay(12, 25);  % Initialize GammaDelay object
        end

        function init(self)
            % Initialization method
            % Initializes the GammaDelay object
            
            self.hGammaDelay.init();
        end

        function mi1Opt = process(self, tm3Ipt)
            % Processing method
            % Applies the delay mechanism to the input matrix tm3Ipt
            
            mi1Opt = self.hGammaDelay.process(tm3Ipt);
            self.Opt = mi1Opt;
        end
    end

end
