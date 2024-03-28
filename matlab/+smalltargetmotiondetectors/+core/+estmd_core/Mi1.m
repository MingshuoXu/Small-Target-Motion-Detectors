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

        function mi1Opt = process(self, tm3Signal)
            % Processing method

            % Apply gamma delay to the input signal
            if iscell(tm3Signal)
                mi1Opt = self.hGammaDelay.process_cell(tm3Signal);
            else
                mi1Opt = self.hGammaDelay.process(tm3Signal);
            end

            self.Opt = mi1Opt;
        end
    end

end
