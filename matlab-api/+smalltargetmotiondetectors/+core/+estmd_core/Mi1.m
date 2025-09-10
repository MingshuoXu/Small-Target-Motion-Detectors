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
            import smalltargetmotiondetectors.core.math_operator.*;

            % Initialize GammaDelay object
            self.hGammaDelay = GammaDelay(12, 25);  
        end

        function init_config(self)
            % Initialization method
            % Initializes the GammaDelay object
            
            self.hGammaDelay.init_config();
        end

        function mi1Opt = process(self, tm3Signal)
            % Processing method

            % Apply gamma delay to the input signal
            mi1Opt = self.hGammaDelay.process(tm3Signal);
            
            self.Opt = mi1Opt;
        end
    end

end
