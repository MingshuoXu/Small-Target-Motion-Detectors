classdef Medulla < smalltargetmotiondetectors.core.estmd_backbone.Medulla
    % Medulla layer of the motion detection system
    %   This class represents the Medulla layer of the motion detection
    %   system. It contains components such as Tm1, Tm2, Tm3, and Mi1.


    properties
        hDelay5Mi1;
        hDelay5Tm1;
    end

    properties(Hidden)
        cellTm1Ipt;
    end


    methods
        function self = Medulla()
            % Constructor method
            
            import smalltargetmotiondetectors.core.*;
            
            % Initializes the Medulla object
            self = self@...
                smalltargetmotiondetectors.core.estmd_backbone.Medulla();

            self.hTm1 = GammaDelay(5,25);
            self.hDelay5Mi1 = GammaDelay(25,30);
            self.hDelay5Tm1 = GammaDelay(25,30);

            self.cellTm1Ipt = CellRecording();
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Medulla layer components
            init@smalltargetmotiondetectors.core.estmd_backbone.Medulla(self);
            self.hTm1.init(false);
            self.hDelay5Mi1.init();
            self.hDelay5Tm1.init(false);

            if isempty(self.cellTm1Ipt.lenCell)
                self.cellTm1Ipt.lenCell = max(...
                    self.hDelay5Mi1.lenKernel, ...
                    self.hDelay5Tm1.lenKernel);
            end
            self.cellTm1Ipt.init();
        end

        function varageout = process(self, MedullaIpt)
            % Processing method
            % Applies processing to the input and returns the output
            
            % Process Tm2 and Tm3 components
            tm2Signal = self.hTm2.process(MedullaIpt);
            tm3Signal = self.hTm3.process(MedullaIpt);

            % Process Tm1 component using output of Tm2
            self.cellTm1Ipt.push(tm3Signal);
            tm1Delay3Signal = self.hTm1.process_cell(self.cellTm1Ipt.cellData);
            tm1Delay5Signal = self.hDelay5Tm1.process_cell(self.cellTm1Ipt.cellData);

            % Process Mi1 component using output of Tm3
            mi1Delay5Signal = self.hDelay5Mi1.process(tm3Signal);
            
            % Store the output signals in Opt property
            varageout = {...
                tm3Signal, ...
                tm1Delay3Signal, ...
                mi1Delay5Signal, ...
                tm2Signal, ...
                tm1Delay5Signal};
            self.Opt = varageout;
        end
    end

end
