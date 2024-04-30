classdef Mi1 < smalltargetmotiondetectors.core.BaseCore
    % Mi1 class for motion detection
    %   This class represents the Mi1 neuron in the DSTMD. 
    % It applies a gamma delay to the input signal.
    %

    properties
        hGammaDelay; % Gamma delay component
    end

    methods
        function self = Mi1()
            % Constructor method
            % Initializes the Mi1 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.math_operator.GammaDelay;

            % Initialize gamma delay component with default parameters
            self.hGammaDelay = GammaDelay(3, 15);
        end

        function init_config(self)
            % Initialization method
            % Initializes the gamma delay component
            
            self.hGammaDelay.init_config();
        end

        function mi1Opt = process(self, tm3Signal)
            % Processing method
            % Processes the input signal using the gamma delay component

            % Apply gamma delay to the input signal
            mi1Opt = self.hGammaDelay.process(tm3Signal);

            % Set output
            self.Opt = mi1Opt;
        end
    end

end
