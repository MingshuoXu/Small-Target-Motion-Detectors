classdef Lobula < smalltargetmotiondetectors.core.backbonev2_core.Lobula
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
            
            self = self@smalltargetmotiondetectors.core.backbonev2_core.Lobula();
            import smalltargetmotiondetectors.core.*;
            import smalltargetmotiondetectors.core.backbonev2_core.*;
            
            % Initialize the SurroundInhibition component
            self.hSTMD = smalltargetmotiondetectors.core.stfeedbackstmd_core.Stmdcell();
            self.hLPTC = smalltargetmotiondetectors.core.stfeedbackstmd_core.Lptcell();

            self.hGammaDelay = GammaDelay(6,12);

        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Lobula layer component
            init@smalltargetmotiondetectors.core.backbonev2_core.Lobula(self);

            self.hSTMD.init();
            self.hLPTC.init(self.hSTMD.hGammaDelay.lenKernel);
        end

        function [nmsOpt, direciton] = process(self, varagein)
            % Processing method
            % Performs a correlation operation on the ON and OFF channels
            % and then applies surround inhibition

            % Extract ON and OFF channel signals from the input
            onSignal = varagein{1};
            offSignal = varagein{2};

            [psi, fai] = self.hLPTC.process(...
                offSignal, offSignal, onSignal, onSignal);

            lobulaOpt = ...
                self.hSTMD.process(onSignal, offSignal, psi, fai);
            
            [nmsOpt, direciton] = self.hDireCell.process(lobulaOpt);

            % Store the output in Opt property
            self.Opt = {nmsOpt,direciton};
        end
    end

end












