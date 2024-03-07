classdef Tm3 < smalltargetmotiondetectors.core.BaseCore
    % Tm3 class for motion detection
    %   This class represents the Tm3 neuron in the DSTMD.
    %   It processes the input matrix by performing a maximum operation
    %   with zero for negative values.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    methods
        function self = Tm3()
            % Constructor method
            % Initializes the Tm3 object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method is not used in this class
        end

        function tm3Opt = process(self, iptMatrix)
            % Processing method
            % Processes the input matrix by performing a maximum
            % operation with zero for negative values
            
            tm3Opt = max(iptMatrix, 0);
            self.Opt = tm3Opt; % Update output (assuming 'self.Opt' should be assigned)
        end
    end

end
