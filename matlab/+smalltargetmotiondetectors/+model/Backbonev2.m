classdef Backbonev2 < smalltargetmotiondetectors.model.BaseModel
    %SpikingSTMD_Core SNN in STMD based model


    properties
    end
    properties(Hidden)

    end

    methods
        % Constructor function
        function self = Backbonev2()
            self@smalltargetmotiondetectors.model.BaseModel();

            import smalltargetmotiondetectors.core.estmd_core.*;
            import smalltargetmotiondetectors.core.fracstmd_core.*;
            import smalltargetmotiondetectors.core.backbonev2_core.*;

            self.hRetina = ...
                smalltargetmotiondetectors.core.estmd_core.Retina();
            self.hLamina = ...
                smalltargetmotiondetectors.core.fracstmd_core.Lamina();
            self.hMedulla = ...
                smalltargetmotiondetectors.core.backbonev2_core.Medulla();
            self.hLobula = ...
                smalltargetmotiondetectors.core.backbonev2_core.Lobula();
            
            self.hLamina.alpha = 0.3;
        end

        function init_config(self)
            self.hRetina.init_config();
            self.hLamina.init_config();
            self.hMedulla.init_config();
            self.hLobula.init_config();
        end

        function model_structure(self, modelIpt)
            

            self.retinaOpt = self.hRetina.process(modelIpt);
            self.laminaOpt = self.hLamina.process(self.retinaOpt);
            %             self.medullaOpt = ...
            %                 self.hMedulla.process(self.laminaOpt, self.isSpike);
            %             [self.lobulaOpt, self.isSpike] = ...
            %                 self.hLobula.process(self.medullaOpt);

            self.hMedulla.process(self.laminaOpt);
            self.medullaOpt = self.hMedulla.Opt;

            [self.lobulaOpt, self.modelOpt.direction] = ...
                self.hLobula.process(self.medullaOpt);

            self.modelOpt.response = self.lobulaOpt;
        end


    end % end methods

end

