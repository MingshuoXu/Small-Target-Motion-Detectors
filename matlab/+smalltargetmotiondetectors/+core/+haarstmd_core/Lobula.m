classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of Haar STMD method
    
    properties
        tau = 1; % a parameter to align the spacialOpt and temporalOpt
    end
    properties(Hidden)
        hSubInhi;  % SurroundInhibition component
    end
    
    
    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.math_operator.*;
            
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();
            self.hSubInhi.B = 1;
        end
        
        function init_config(self)
            % Initialization method
            self.hSubInhi.init_config();
        end
        
        function lobulaOpt = process(self, listSpatialOpt, temporalOpt)
            % Processing method
            % Performs a correlation operation on the spatial output and
            % temporal output, and then applies surround inhibition
            
            % Perform the correlation operation
            point = mod(listSpatialOpt.point-self.tau, listSpatialOpt.len) + 1; 
            spatialOpt = listSpatialOpt.data{point};
            
            if ~isempty(spatialOpt)
                correlationOutput = spatialOpt .* temporalOpt;
            else
                correlationOutput = zeros(size(temporalOpt));
            end
            
            % Apply surround inhibition           
            lobulaOpt = max(self.hSubInhi.process(correlationOutput) , 0);
            
            % Store the output in Opt property
            self.Opt = lobulaOpt;
        end % [EoF]

    end % [EoM]
    
end % [EoC]












