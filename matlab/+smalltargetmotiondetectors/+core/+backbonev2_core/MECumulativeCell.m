classdef MECumulativeCell < smalltargetmotiondetectors.core.BaseCore
    %umulativeCell cell in Medulla layer 
    %
    properties
        coeffDecay = 0.5; % coefficient of decay
    end
    
    properties(Hidden)
        postMP;
    end
    
    properties(Constant)
        V_REST = 0;  % passive/rest potentials;
        V_EXCI = 1;  % excitatory saturation potentials;
    end

    methods
        function self = MECumulativeCell()
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end

        function init_config(self)
            return;
        end

        function varargout = process(self, samePolarity, oppoPolarity)
            if isempty(self.postMP)
                self.postMP = zeros(size(samePolarity));
            end

            % Decay
            decayTerm = self.coeffDecay * (self.V_REST - self.postMP);
            % Inhibition
            %   inhiGain = exp(oppoPolarity);
            % To reduce the computational load, a secone-order Taylor 
            %   expansion was used.
            inhiGain = 1 + oppoPolarity + oppoPolarity.*2;
            
            % Excitation
            exciTerm = samePolarity .* (self.V_EXCI - self.postMP);
            
            % Euler method for solving ordinary differential equation
            self.postMP = self.postMP + inhiGain .* decayTerm + exciTerm;
            
            % Output
            varargout = {self.postMP};
        end

    end

end
















