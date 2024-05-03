classdef Tm2 < smalltargetmotiondetectors.core.backbonev2_core.MedullaCell
    %Tm2 


    methods
        function self = Tm2()
            self = ...
                self@smalltargetmotiondetectors.core.backbonev2_core.MedullaCell();

        end
    end

    methods
        function init_config(self)
            
        end

        function tm2Opt = process(self, tm2Ipt)%, IsLastSpike)

            offSignal = max(-tm2Ipt,0);
            tm2Opt = ...
                process@smalltargetmotiondetectors.core.backbonev2_core.MedullaCell(...
                self, offSignal); %, IsLastSpike);
            self.Opt = tm2Opt;

        end

    end

end
















