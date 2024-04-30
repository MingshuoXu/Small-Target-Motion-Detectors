classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs a correlation operation on the ON and OFF
    %   channels and then applies surround inhibition.

    properties
        hSTMD;
        hLPTC;
    end


    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.*;
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;
            
            % Initialize the SurroundInhibition component
            self.hSTMD = ...
                smalltargetmotiondetectors.core.stfeedbackstmd_core.Stmdcell();
            self.hLPTC = ...
                smalltargetmotiondetectors.core.stfeedbackstmd_core.Lptcell();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            self.hSTMD.init_config();
            self.hLPTC.init_config(self.hSTMD.hGammaDelay.lenKernel);
        end

        function lobulaOpt = process(self, varagein)
            % Processing method
            % Performs a correlation operation on the ON and OFF channels
            % and then applies surround inhibition
            
            % Extract ON and OFF channel signals from the input
            [tm3Signal, ...
            tm1Para3Signal, ...
            mi1Para5Signal, ...
            tm2Signal, ...
            tm1Para5Signal, ...
            tau5] = deal(varagein{:});
            % tm1Signal = varagein{1};
            % tm2Signal = varagein{2};
            % tm3Signal = varagein{3};
            % mi1Signal = varagein{4};


            [psi, fai] = self.hLPTC.process(...
                tm3Signal, ...
                mi1Para5Signal, ...
                tm2Signal, ...
                tm1Para5Signal, ...
                tau5);

            lobulaOpt = ...
                self.hSTMD.process(tm3Signal, tm1Para3Signal, psi, fai);
            
            % Store the output in Opt property
            self.Opt = lobulaOpt;
        end
    end

end












