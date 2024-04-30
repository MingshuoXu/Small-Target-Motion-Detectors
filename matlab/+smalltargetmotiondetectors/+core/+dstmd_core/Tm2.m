classdef Tm2 < smalltargetmotiondetectors.core.BaseCore
    % Tm2 class for motion detection
    %   This class represents the Tm2 neuron in the medulla layer of the
    %   DSTMD (Directional-Small Target Motion Detector).
    %   It processes the input matrix by performing a maximum operation
    %   with zero for negative values.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    methods
        function self = Tm2()
            % Constructor method
            % Initializes the Tm2 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method is not used in this class
        end

        function tm2Opt = process(self, iptMatrix)
            % Processing method
            % Processes the input matrix by performing a maximum
            % operation with zero for negative values
            
            tm2Opt = max(-iptMatrix, 0);
            self.Opt = tm2Opt; % Update output (assuming 'self.Opt' should be assigned)
        end
    end

end
