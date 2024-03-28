classdef Tm1 < smalltargetmotiondetectors.core.BaseCore
    % Tm1 class for motion detection
    %   This class represents the Tm1 neuron in the DSTMD.
    % It applies a gamma delay to the input signal.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        hGammaDelay; % Gamma delay component
    end

    methods
        function self = Tm1()
            % Constructor method
            % Initializes the Tm1 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.GammaDelay;

            % Initialize gamma delay component with default parameters
            self.hGammaDelay = GammaDelay(5, 25);
        end

        function init(self)
            % Initialization method
            % Initializes the gamma delay component
            
            self.hGammaDelay.init();
        end

        function tm1Opt = process(self, tm2Signal)
            % Processing method
            % Processes the input signal using the gamma delay component
            
            % Apply gamma delay to the input signal
            if iscell(tm2Signal)
                tm1Opt = self.hGammaDelay.process_cell(tm2Signal);
            else
                tm1Opt = self.hGammaDelay.process(tm2Signal);
            end
            
            % Set output
            self.Opt = tm1Opt;
        end

    end

end
