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
        hMi1Para4; % Delayed ON signal component for mi1 with n4 and tau4

        hTm2; % OFF signal component
        hTm1Para5; % Delayed OFF signal component for tm1 with n5 and tau5
        hTm1Para6; % Delayed OFF signal component for tm1 with n6 and tau6
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
            import smalltargetmotiondetectors.util.CircularCell;
            

            % Initialize components
            self.hTm3 = Tm3();
            self.hMi1Para4 = Mi1();

            self.hTm2 = Tm2();
            self.hTm1Para5 = Tm1();
            self.hTm1Para6 = Tm1();
            
            % Set parameters for hDelay6Tm1
            self.hTm1Para6.hGammaDelay.order = 8;
            self.hTm1Para6.hGammaDelay.tau = 40;

            self.cellTm1Ipt = CircularCell();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % Initializes the delay components
            
            self.hTm1Para5.hGammaDelay.isRecord = false;
            self.hTm1Para6.hGammaDelay.isRecord = false;

            self.hMi1Para4.init_config();
            self.hTm1Para5.init_config();
            self.hTm1Para6.init_config();

            if isempty(self.cellTm1Ipt.len)
            self.cellTm1Ipt.len = max( ...
                self.hTm1Para5.hGammaDelay.lenKernel, ...
                self.hTm1Para6.hGammaDelay.lenKernel);
            end
            self.cellTm1Ipt.init_config();
        end

        function varargout = process(self, MedullaInput)          

            % Processing method
            % Processes input signals and produces output
            
            % Process Tm3 and Tm2 signals
            tm3Signal = self.hTm3.process(MedullaInput);
            tm2Signal = self.hTm2.process(MedullaInput);
            
            % Process signals with delays
            mi1Para4Signal = self.hMi1Para4.process(tm3Signal);
            
            self.cellTm1Ipt.circrecord(tm2Signal);
            tm1Para5Signal = self.hTm1Para5.process( ...
                self.cellTm1Ipt);
            tm1Para6Signal = self.hTm1Para6.process( ...
                self.cellTm1Ipt);

            % Output signals
            varargout = {...
                tm3Signal, ...
                mi1Para4Signal, ...
                tm1Para5Signal, ...
                tm1Para6Signal};
            self.Opt = varargout;
        end
    end
end
