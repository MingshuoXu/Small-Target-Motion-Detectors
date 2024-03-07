classdef Lamina < smalltargetmotiondetectors.core.BaseCore
    % Lamina class for the lamina layer
    %   This class implements the lamina layer in the FracSTMD
    
    properties
        alpha = 0.8;
        fps = 240;
        delta = 20;
    end
    
    properties(Hidden)
        fracKernel;
        paraCur;
        paraPre;
        preLaminaIpt = [];
        preLaminaOpt = [];
    end
    
    methods
        function self = Lamina()
            % Constructor method
            % Initializes the Lamina object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end
    
    methods
        function init(self)
            % Initialization method
            % Initializes the fractional differential kernel
            
            import smalltargetmotiondetectors.tool.kernel.*;
            
            self.fracKernel = create_fracdiff_kernel(self.alpha, self.delta);
            
            self.paraCur = self.fracKernel(1);
            self.paraPre = exp(-self.alpha / (1 - self.alpha));
        end
        
        function laminaOpt = process(self, LaminaIpt)
            % Processing method
            % Processes the LaminaIpt to generate the lamina output
            
            if isempty(self.preLaminaIpt)
                diffLaminaIpt = zeros(size(LaminaIpt));
            else
                % First order difference
                diffLaminaIpt = LaminaIpt - self.preLaminaIpt;
            end
            
            % laminaOpt = compute_by_conv(self, diffLaminaIpt)
            laminaOpt = compute_by_iteration(self, diffLaminaIpt);
            
            self.Opt = laminaOpt;
            
            self.preLaminaIpt = LaminaIpt;
        end
    end
    
    methods
        % The compute_by_iteration function is similar but not equivalent
        % to the compute_by_conv function. Since compute_by_iteration
        % is faster, we sacrifice some computational precision for efficiency
        function laminaopt = compute_by_conv(self, diffLaminaIpt)
            import smalltargetmotiondetectors.tool.compute.*;
            self.cellRetinaOutput(1) = [];
            self.cellRetinaOutput{end+1} = diffLaminaIpt;
            
            laminaopt = compute_temporal_conv( ...
                self.cellRetinaOutput, ...
                self.fracKernel ...
                );
        end
        
        function laminaopt = compute_by_iteration(self, diffLaminaIpt)
            if isempty(self.preLaminaOpt)
                laminaopt = self.paraCur * diffLaminaIpt;
            else
                laminaopt = self.paraCur * diffLaminaIpt ...
                    + self.paraPre * self.preLaminaOpt;
            end
            
            self.preLaminaOpt = laminaopt;
        end
    end
end
