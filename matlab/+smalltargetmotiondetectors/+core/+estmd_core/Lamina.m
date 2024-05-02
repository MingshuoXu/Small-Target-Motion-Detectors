classdef Lamina < smalltargetmotiondetectors.core.BaseCore
    % LAMINA Lamina layer
    %   This class implements the Lamina layer of the ESTMD
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    
    properties
        % Handle to the GammaBankPassFilter object
        hGammaBandPassFilter;       
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
            
            self.hGammaBandPassFilter = GammaBandPassFilter();
            self.hLaminaLateralInhibition = LaminaLateralInhibition();
        end
    end
    
    methods
        function init_config(self)
            % Initialization method
            % Initializes the GammaBankPassFilter and LaminaLateralInhibition objects
            
            self.hGammaBandPassFilter.init_config();
            self.hLaminaLateralInhibition.init_config();
        end
        
        function laminaOpt = process(self, laminaIpt)
            % Processing method
            % Applies GammaBankPassFilter and LaminaLateralInhibition to the input matrix
            
            signalWithBPF = self.hGammaBandPassFilter.process(laminaIpt);
            laminaOpt = self.hLaminaLateralInhibition.process(signalWithBPF);
            self.Opt = laminaOpt;
        end
    end
end
