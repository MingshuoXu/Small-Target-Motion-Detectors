classdef Mi4 < smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell
%Mi4 Mi4 cell of ON pathway in the Medulla layer.

    methods
        function self = Mi4()
            self = self@smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell();
        end

        function init_config(self)
            return;
        end

        function optSignal = process(self, iptMatrix)
            onSignal = max(iptMatrix, 0);
            optSignal = ...
                process@smalltargetmotiondetectors.core.backbonev2_core.MECumulativeCell(...
                self, onSignal); 
            self.Opt = optSignal;
        end
        
    end

end
















