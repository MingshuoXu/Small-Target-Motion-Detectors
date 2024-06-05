classdef MECumulativeCell < smalltargetmotiondetectors.core.BaseCore
    %umulativeCell cell in Medulla layer 
    %
    properties
        coeffDecay = 0.2;
        coeffInhi = 3;
        postMP;
    end

    methods
        function self = MECumulativeCell()
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init_config(self)
            return;
        end

        function varargout = process(self, samePolarity, oppoPolarity)
            if isempty(self.postMP)
                self.postMP = zeros(size(samePolarity));
            end

            % Decay
            NegativeChange = self.coeffDecay * self.postMP;
            
            % Inhibition
            isInhi = (oppoPolarity>0);
            NegativeChange(isInhi) = self.coeffInhi * NegativeChange(isInhi);
            
            % Excitation
            self.postMP = self.postMP - NegativeChange + samePolarity;
            
            varargout = {self.postMP};
        end

    end

end
















