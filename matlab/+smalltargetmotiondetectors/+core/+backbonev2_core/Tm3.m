classdef Tm3 < smalltargetmotiondetectors.core.backbonev2_core.MedullaCell
    %Tm3 
    %   



    methods
        function self = Tm3()
            self = self@smalltargetmotiondetectors.core.backbonev2_core.MedullaCell();

        end

        function init_config(self)
            
        end

        function tm3Opt = process(self, tm3Ipt)%, IsLastSpike)

            onSignal = max(tm3Ipt, 0);
            tm3Opt = ...
                process@smalltargetmotiondetectors.core.backbonev2_core.MedullaCell(...
                self, onSignal); %, IsLastSpike);
            self.Opt = tm3Opt;
        end



    end

end
















