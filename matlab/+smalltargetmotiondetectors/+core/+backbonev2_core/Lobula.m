classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    
    properties
        hSubInhi;
        hLPTC;
    end


    methods
        function self = Lobula()
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.math_operator.*;
            import smalltargetmotiondetectors.core.backbonev2_core.*;

            self.hSubInhi = SurroundInhibition();
            self.hLPTC = ...
                smalltargetmotiondetectors.core.backbonev2_core.LPTangentialCell();
        end

        function init_config(self)
            self.hSubInhi.init_config();
            self.hLPTC.init_config();
        end

        function varargout = process(self, onSignal, offSignal, laminaOpt)
            if nargout > 1
                direciton = self.hLPTC.process(laminaOpt, onSignal, offSignal);
            end
            
            correlationOutput = onSignal .* offSignal;
            lobulaOpt = self.hSubInhi.process(correlationOutput);            
            
            if nargout == 1
                varargout = {inhiOpt}; 
            elseif nargout == 2               
                varargout = {lobulaOpt, direciton};
            elseif nargout == 3
                varargout = {lobulaOpt, direciton, correlationOutput};
            end
            
            self.Opt = varargout;
        end

    end

end
















