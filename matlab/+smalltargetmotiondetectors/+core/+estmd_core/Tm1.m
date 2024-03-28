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

            % Apply gamma delay to the input signal
            if iscell(tm2Signal)
                tm1Opt = self.hGammaDelay.process_cell(tm2Signal);
            else
                tm1Opt = self.hGammaDelay.process(tm2Signal);
            end

            self.Opt = tm1Opt;
        end
    end

end
