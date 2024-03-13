classdef GammaBankPassFilter < smalltargetmotiondetectors.core.BaseCore
    % GammaBankPassFilter Gamma bank pass filter
    %   This class implements a gamma bank pass filter with two gamma
    %   delays.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        hGammaDelay1;  % Gamma delay 1 object
        hGammaDelay2;  % Gamma delay 2 object
    end

    methods
        function self = GammaBankPassFilter()
            % Constructor method
            % Initializes the GammaBankPassFilter object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.GammaDelay;
            
            % Initialize gamma delays with specified parameters
            self.hGammaDelay1 = GammaDelay(2, 3);
            self.hGammaDelay2 = GammaDelay(3, 6);
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes gamma delays
            if self.hGammaDelay1.tau < self.hGammaDelay2.tau
                self.hGammaDelay2.tau = self.hGammaDelay1.tau + 1;
            end

            self.hGammaDelay1.init();  % Initialize gamma delay 1
            self.hGammaDelay2.init();  % Initialize gamma delay 2
        end

        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Processes the input matrix using gamma delays
            
            % Compute outputs of gamma delays
            gamma1Output = self.hGammaDelay1.process(iptMatrix);
            gamma2Output = self.hGammaDelay2.process(iptMatrix);
            
            % Compute the difference between the outputs
            optMatrix = gamma1Output - gamma2Output;
        end
    end

end
