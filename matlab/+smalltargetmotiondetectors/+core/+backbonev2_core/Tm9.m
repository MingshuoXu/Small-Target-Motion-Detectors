classdef Tm9 < smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell
%Tm9 Tm9 cell of OFF pathway in the Medulla layer.

    methods
        function self = Tm9()
            self = ...
                self@smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell();

        end
    end

    methods
        function init_config(self)
            return;
        end

        function optSignal = process(self, iptMatrix)

            offSignal = max(-iptMatrix,0);
            optSignal = ...
                process@smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell(...
                self, offSignal); 
            self.Opt = optSignal;

        end

    end

end
















