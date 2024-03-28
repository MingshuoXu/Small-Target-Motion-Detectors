classdef Medulla < smalltargetmotiondetectors.core.BaseCore
    % Medulla class for motion detection
    %   This class represents the medulla layer in the motion detection
    %   system. It processes inputs from Tm3, Tm2, Mi1, and Tm1 to produce
    %   output signals.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        

        hTm3; % ON signal component
        hDelay4Mi1; % Delayed ON signal component

        hTm2; % OFF signal component
        hDelay5Tm1; % Delayed OFF signal component for tm1
        hDelay6Tm1; % Delayed OFF signal component for tm1 with different parameters
    end

    properties(Hidden)
        cellTm1Ipt; % Medulla input recorded in cell
    end


    methods
        function self = Medulla()
            % Constructor method
            % Initializes the Medulla object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();

            import smalltargetmotiondetectors.core.*;
            import smalltargetmotiondetectors.core.dstmd_core.*;
            

            % Initialize components
            self.hTm3 = Tm3();
            self.hDelay4Mi1 = Mi1();

            self.hTm2 = Tm2();
            self.hDelay5Tm1 = Tm1();
            self.hDelay6Tm1 = Tm1();
            
            % Set parameters for hDelay6Tm1
            self.hDelay6Tm1.hGammaDelay.order = 8;
            self.hDelay6Tm1.hGammaDelay.tau = 40;

            self.cellTm1Ipt = CellRecording();
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes the delay components
            
            self.hDelay5Tm1.hGammaDelay.isRecord = false;
            self.hDelay6Tm1.hGammaDelay.isRecord = false;

            self.hDelay4Mi1.init();
            self.hDelay5Tm1.init();
            self.hDelay6Tm1.init();

            if isempty(self.cellTm1Ipt.lenCell)
            self.cellTm1Ipt.lenCell = max( ...
                self.hDelay5Tm1.hGammaDelay.lenKernel, ...
                self.hDelay6Tm1.hGammaDelay.lenKernel);
            end
            self.cellTm1Ipt.init();
        end

        function varargout = process(self, MedullaInput)          

            % Processing method
            % Processes input signals and produces output
            
            

            % Process Tm3 and Tm2 signals
            tm3Signal = self.hTm3.process(MedullaInput);
            tm2Signal = self.hTm2.process(MedullaInput);
            
            % Process signals with delays
            mi1Delay4Signal = self.hDelay4Mi1.process(tm3Signal);
            
            self.cellTm1Ipt.push(tm2Signal);
            tm1Delay5Signal = ...
                self.hDelay5Tm1.process(self.cellTm1Ipt.cellData);
            tm1Delay6Signal = ...
                self.hDelay6Tm1.process(self.cellTm1Ipt.cellData);

            % Output signals
            varargout = {...
                tm3Signal, ...
                mi1Delay4Signal, ...
                tm1Delay5Signal, ...
                tm1Delay6Signal};
            self.Opt = varargout;
        end
    end
end
