classdef Lamina < smalltargetmotiondetectors.core.BaseCore
    % Lamina layer of the motion detection system
    %   This class represents the Lamina layer of the motion detection
    %   system. It applies gamma bank pass filtering to the input.

    properties
        hGammaBankPassFilter;  % GammaBankPassFilter component
    end

    methods
        function self = Lamina()
            % Constructor method
            % Initializes the Lamina object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.math_operator.*;

            % Initialize the GammaBankPassFilter component
            self.hGammaBankPassFilter = GammaBandPassFilter();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes the Lamina layer component
            
            self.hGammaBankPassFilter.init_config();
        end

        function laminaOpt = process(self, laminaIpt)
            % Processing method
            % Applies gamma bank pass filtering to the input
            
            % Process the input using GammaBankPassFilter
            laminaOpt = self.hGammaBankPassFilter.process(laminaIpt);
            
            % Store the output in Opt property
            self.Opt = laminaOpt;
        end
    end

end
