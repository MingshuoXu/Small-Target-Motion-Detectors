classdef Tm1 < smalltargetmotiondetectors.core.BaseCore
    % Tm1 
    
    properties
        hGammaDelay;  % Handle to the GammaDelay object
    end

    methods
        function self = Tm1()
            % Constructor method
            % Initializes the Tm1 object
            
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

        function tm1Opt = process(self, tm2Signal)
            % Processing method

            % Apply gamma delay to the input signal
            tm1Opt = self.hGammaDelay.process(tm2Signal);

            self.Opt = tm1Opt;
        end
    end

end
