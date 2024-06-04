classdef MECumulativeCell < smalltargetmotiondetectors.core.BaseCore
    %umulativeCell cell in Medulla layer 
    %
    properties
        coeffDecay = 0.8;
        coeffInhi = 0.5;
        postMembranePotential;
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

        function postMP = process(self, iptMatrix)
            if isempty(self.postMembranePotential)
                self.postMembranePotential = zeros(size(iptMatrix));
            end

            % Decay
            postMP = self.coeffDecay * self.postMembranePotential;
            
            % Accumulation
            postMP = postMP + iptMatrix;
            
            % Inhi
            isInhi = (iptMatrix==0);
            postMP(isInhi) = self.coeffInhi * postMP(isInhi);

            self.postMembranePotential = postMP;
        end

    end

end
















