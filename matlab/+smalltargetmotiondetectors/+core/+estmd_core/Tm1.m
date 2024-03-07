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
            import smalltargetmotiondetectors.core.GammaDelay;
            
            self.hGammaDelay = GammaDelay(12, 25);  % Initialize GammaDelay object
        end

        function init(self)
            % Initialization method
            % Initializes the GammaDelay object
            
            self.hGammaDelay.init();
        end

        function tm1Opt = process(self, tm2Signal)
            % Processing method
            % Applies the delay mechanism to the input matrix tm2Signal
            
            tm1Opt = self.hGammaDelay.process(tm2Signal);
            self.Opt = tm1Opt;
        end
    end

end
