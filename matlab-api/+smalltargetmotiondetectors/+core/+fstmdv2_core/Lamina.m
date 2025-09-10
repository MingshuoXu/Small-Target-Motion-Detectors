classdef Lamina < smalltargetmotiondetectors.core.fracstmd_core.Lamina
    % Lamina class for the lamina layer
    %   This class implements the lamina layer in the FracSTMD
    
    properties(Hidden)
        loopLaminaOpt;
    end
    
    properties(Hidden)
        isInLoop;
    end
    
    methods
        function self = Lamina()
            % Constructor method
            % Initializes the Lamina object
            
            self = ...
                self@smalltargetmotiondetectors.core.fracstmd_core.Lamina();
        end
    end
    
    methods
        
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
            
            if ~self.isInLoop
                self.preLaminaIpt = LaminaIpt;
            end
        end
    end
    
    methods
        
        function laminaopt = compute_by_iteration(self, diffLaminaIpt)
            if isempty(self.preLaminaOpt)
                laminaopt = self.paraCur * diffLaminaIpt;
            else
                if ~self.isInLoop
                    self.preLaminaIpt = self.loopLaminaOpt;
                end
                laminaopt = self.paraCur * diffLaminaIpt ...
                    + self.paraPre * self.preLaminaOpt;
            end
            
             self.loopLaminaOpt = laminaopt;
             
        end
    end
end
