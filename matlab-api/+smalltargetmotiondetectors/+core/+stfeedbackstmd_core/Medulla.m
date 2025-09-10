classdef Medulla < smalltargetmotiondetectors.core.estmd_backbone.Medulla
    % Medulla layer of the motion detection system
    %   This class represents the Medulla layer of the motion detection
    %   system. It contains components such as Tm1, Tm2, Tm3, and Mi1.


    properties
        hPara5Mi1;
        hPara5Tm1;
    end

    properties(Hidden)
        cellTm1Ipt;
    end


    methods
        function self = Medulla()
            % Constructor method
            
            import smalltargetmotiondetectors.core.math_operator.GammaDelay;
            import smalltargetmotiondetectors.util.CircularCell;

            % Initializes the Medulla object
            self = self@...
                smalltargetmotiondetectors.core.estmd_backbone.Medulla();

            self.hTm1 = GammaDelay(5,25);
            self.hPara5Mi1 = GammaDelay(25,30);
            self.hPara5Tm1 = GammaDelay(25,30);

            self.cellTm1Ipt = CircularCell();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes the Medulla layer components
            init_config@smalltargetmotiondetectors.core.estmd_backbone.Medulla(self);
            self.hTm1.init_config(false);
            self.hPara5Mi1.init_config();
            self.hPara5Tm1.init_config(false);

            if isempty(self.cellTm1Ipt.len)
                self.cellTm1Ipt.len = max(...
                    self.hPara5Mi1.lenKernel, ...
                    self.hPara5Tm1.lenKernel);
            end
            self.cellTm1Ipt.init_config();
        end

        function varageout = process(self, MedullaIpt)
            % Processing method
            % Applies processing to the input and returns the output
            
            % Process Tm2 and Tm3 components
            tm2Signal = self.hTm2.process(MedullaIpt);
            tm3Signal = self.hTm3.process(MedullaIpt);

            % Process Tm1 component using output of Tm2
            self.cellTm1Ipt.circrecord(tm3Signal);
            tm1Para3Signal = self.hTm1.process(self.cellTm1Ipt);
            tm1Para5Signal = ...
                self.hPara5Tm1.process(self.cellTm1Ipt);

            % Process Mi1 component using output of Tm3
            mi1Para5Signal = self.hPara5Mi1.process(tm3Signal);
            
            % Store the output signals in Opt property
            varageout = {...
                tm3Signal, ...
                tm1Para3Signal, ...
                mi1Para5Signal, ...
                tm2Signal, ...
                tm1Para5Signal, ...
                self.hPara5Mi1.tau};
            self.Opt = varageout;
        end
    end

end
