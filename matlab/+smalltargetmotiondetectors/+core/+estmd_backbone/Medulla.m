classdef Medulla < smalltargetmotiondetectors.core.BaseCore
    % Medulla layer of the motion detection system
    %   This class represents the Medulla layer of the motion detection
    %   system. It contains components such as Tm1, Tm2, Tm3, and Mi1.


    properties
        hTm1;  % Tm1 component
        hTm2;  % Tm2 component
        hTm3;  % Tm3 component
        hMi1;  % Mi1 component
    end

    methods
        function self = Medulla()
            % Constructor method
            % Initializes the Medulla object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.estmd_backbone.*;
            import smalltargetmotiondetectors.core.estmd_core.*;

            % Initialize components
            self.hTm1 = smalltargetmotiondetectors.core.estmd_core.Tm1();
            self.hMi1 = smalltargetmotiondetectors.core.estmd_core.Mi1();
            self.hTm2 = smalltargetmotiondetectors.core.estmd_backbone.Tm2();
            self.hTm3 = smalltargetmotiondetectors.core.estmd_backbone.Tm3();
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Medulla layer components
            
            self.hTm1.init();
            self.hTm2.init();
            self.hTm3.init();
        end

        function varageout = process(self, MedullaIpt)
            % Processing method
            % Applies processing to the input and returns the output
            
            % Process Tm2 and Tm3 components
            tm2Signal = self.hTm2.process(MedullaIpt);
            tm3Signal = self.hTm3.process(MedullaIpt);

            % Process Tm1 component using output of Tm2
            tm1Signal = self.hTm1.process(tm2Signal);
            
            % Store the output signals in Opt property
            varageout = {tm3Signal, tm1Signal};
            self.Opt = varageout;
        end
    end

end
