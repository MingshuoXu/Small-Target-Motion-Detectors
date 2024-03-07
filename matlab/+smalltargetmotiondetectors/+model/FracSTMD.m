classdef FracSTMD < smalltargetmotiondetectors.model.ESTMDBackbone
    % FracSTMD - Fractional-order Small Target Motion Detector
    %   This class implements a fractional-order Small Target Motion Detector
    %   by inheriting from the ESTMDBackbone class.
    
    properties
        % Define properties here (if any)
    end

    properties(Hidden)
        % Define hidden properties here (if any)
    end
    
    
    
    methods
        function self = FracSTMD()
            % FracSTMD Constructor method
            %   Initializes an instance of the FracSTMD class.
            %
            % Syntax:
            %   obj = FracSTMD()
            %
            % Description:
            %   Initializes the FracSTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.ESTMDBackbone();
            % Import necessary packages
            import smalltargetmotiondetectors.core.fracstmd_core.*;

            % Customize Lamina and Lobula components
            self.hLamina = smalltargetmotiondetectors.core.fracstmd_core.Lamina();
            self.hMedulla.hTm1.hGammaDelay.order = 100;
            self.hLobula.hSubInhi.e = 1.8;
        end
    end
end
