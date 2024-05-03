classdef MedullaCell < smalltargetmotiondetectors.core.BaseCore
    %MedullaCell cell in Medulla layer 
    %
    properties
        
        afterhyperpolarizationV = -50;
        coeffDecay = 0.8;
        coeffInhi = 0.5;
    end
    properties%(Hidden)
        Voltage;
        lastInput;
    end

    methods
        function self = MedullaCell()
            self = self@smalltargetmotiondetectors.core.BaseCore();

        end
    end

    methods
        function init_config(self)
            return;
        end

        function Voltage = process(self, iptMatrix)%, IsLastSpike
            if isempty(self.lastInput)
                [m, n] = size(iptMatrix);
                self.Voltage = zeros(m, n);
                self.lastInput = false(m, n);
            end

            % reset V for Spike
%             self.Voltage(IsLastSpike) = self.afterhyperpolarizationV;

            % Decay
            isDecay = self.lastInput; % & ~IsLastSpike;
            self.Voltage(isDecay) = ...
                self.coeffDecay * self.Voltage(isDecay);
            
            % Inhi
            isInhi = ~self.lastInput; % & ~IsLastSpike;
            self.Voltage(isInhi) = self.coeffInhi * self.Voltage(isInhi);

            % Accumulation
            self.Voltage = self.Voltage + iptMatrix;

            % record
            self.lastInput = logical(iptMatrix);

            Voltage = self.Voltage;

        end

    end

end
















