classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs a correlation operation on the ON and OFF
    %   channels and then applies surround inhibition.

    properties
        hSTMD;
        hLPTC;
        hGammaDelay;
    end


    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;
            import smalltargetmotiondetectors.core.*;
            
            % Initialize the SurroundInhibition component
            self.hSTMD = Stmdcell();
            self.hLPTC = Lptcell();

            self.hGammaDelay = GammaDelay(6,12);
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            self.hSTMD.init();
            self.hLPTC.init(self.hSTMD.hGammaDelay.lenKernel);
        end

        function lobulaOpt = process(self, varagein)
            % Processing method
            % Performs a correlation operation on the ON and OFF channels
            % and then applies surround inhibition
            
            % Extract ON and OFF channel signals from the input
            [tm3Signal, ...
            tm1Delay3Signal, ...
            mi1Delay5Signal, ...
            tm2Signal, ...
            tm1Delay5Signal] = deal(varagein{:});
            % tm1Signal = varagein{1};
            % tm2Signal = varagein{2};
            % tm3Signal = varagein{3};
            % mi1Signal = varagein{4};


            [psi, fai] = self.hLPTC.process(...
                tm3Signal, mi1Delay5Signal, tm2Signal, tm1Delay5Signal);

            lobulaOpt = ...
                self.hSTMD.process(tm3Signal, tm1Delay3Signal, psi, fai);
            
            % Store the output in Opt property
            self.Opt = lobulaOpt;
        end
    end

end












