classdef Lamina < smalltargetmotiondetectors.core.BaseCore
    % Lamina class for STMDPlus

    properties
        hGammaBankPassFilter; % Gamma bank-pass filter
    end

    methods
        function self = Lamina()
            % Constructor method
            % Initializes the Lamina object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.estmd_core.GammaBankPassFilter;

            self.hGammaBankPassFilter = GammaBankPassFilter();
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes the hGammaBankPassFilter
            
            self.hGammaBankPassFilter.init();
        end

        function laminaOpt = process(self, laminaIpt)
            % Processing method
            % Processes the input laminaIpt to generate laminaOpt
            
            laminaOpt = self.hGammaBankPassFilter.process(laminaIpt);
            self.Opt = laminaOpt;
        end
    end

end
