classdef Lamina < smalltargetmotiondetectors.core.BaseCore
    % LAMINA Lamina layer
    %   This class implements the Lamina layer of the ESTMD
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    
    properties
        % Handle to the GammaBankPassFilter object
        hGammaBankPassFilter;       
        % Handle to the LaminaLateralInhibition object
        hLaminaLateralInhibition;   
    end
    
    methods
        function self = Lamina()
            % Constructor
            % Initializes the Lamina object and creates GammaBankPassFilter
            % and LaminaLateralInhibition objects
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.estmd_core.*;
            import smalltargetmotiondetectors.core.math_operator.*;
            
            self.hGammaBankPassFilter = GammaBankPassFilter();
            self.hLaminaLateralInhibition = LaminaLateralInhibition();
        end
    end
    
    methods
        function init_config(self)
            % Initialization method
            % Initializes the GammaBankPassFilter and LaminaLateralInhibition objects
            
            self.hGammaBankPassFilter.init_config();
            self.hLaminaLateralInhibition.init_config();
        end
        
        function laminaOpt = process(self, laminaIpt)
            % Processing method
            % Applies GammaBankPassFilter and LaminaLateralInhibition to the input matrix
            
            signalWithBPF = self.hGammaBankPassFilter.process(laminaIpt);
            laminaOpt = self.hLaminaLateralInhibition.process(signalWithBPF);
            self.Opt = laminaOpt;
        end
    end
end
