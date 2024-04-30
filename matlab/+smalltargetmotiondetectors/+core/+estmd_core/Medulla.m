classdef Medulla < smalltargetmotiondetectors.core.BaseCore
    % Medulla Layer of DSTMD
    %   This class implements the Medulla layer of the ESTMD.

    
    properties
        hTm1;  % Handle to the Tm1 object
        hTm2;  % Handle to the Tm2 object
        hTm3;  % Handle to the Tm3 object
    end

    methods
        function self = Medulla()
            % Constructor method
            % Initializes the Medulla object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.estmd_core.*;

            self.hTm1 = Tm1();  % Initialize Tm1 object
            self.hTm2 = Tm2();  % Initialize Tm2 object
            self.hTm3 = Tm3();  % Initialize Tm3 object
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % Initializes the Tm1, Tm2, and Tm3 objects
            
            self.hTm1.init_config();
            self.hTm2.init_config();
            self.hTm3.init_config();
        end

        function varargout = process(self, MedullaIpt)
            % Processing method
            % Processes the input MedullaIpt through Tm1, Tm2, and Tm3 layers
            
            tm2Signal = self.hTm2.process(MedullaIpt);  % Process input through Tm2
            tm3Signal = self.hTm3.process(MedullaIpt);  % Process input through Tm3

            tm1Signal = self.hTm1.process(tm2Signal);  % Process Tm2 output through Tm1

            % Output Tm3 and Tm1 signals
            varargout(1) = {tm3Signal}; 
            varargout(2) = {tm1Signal};  
            self.Opt = varargout;  % Update Opt property with output
        end
    end

end
