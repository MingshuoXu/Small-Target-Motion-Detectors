classdef ESTMD < smalltargetmotiondetectors.model.BaseModel
    % ESTMD - Extended Small Target Motion Detector
    %   This class implements an Extended Small Target Motion Detector
    %   by inheriting from the BaseModel class.
    %
    % Ref: S.D. Wiederman, P.A. Shoemarker, D.C. O Carroll, A model for the
    % detection of moving targets in visual clutter inspired by insect 
    % physiology, PLoS ONE 3 (7) (2008) e2784.
    % (the first STMD-based model)
    properties
        % Define properties here (if any)
    end
     
    methods
        function self = ESTMD()
            % ESTMD Constructor method
            %   Initializes an instance of the ESTMD class.
            %
            % Syntax:
            %   obj = ESTMD()
            %
            % Description:
            %   Initializes the ESTMD object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.BaseModel();
            
            % Import necessary packages
            import smalltargetmotiondetectors.core.estmd_core.*;
            

            % Initialize components
            self.hRetina = ...
                smalltargetmotiondetectors.core.estmd_core.Retina();
            self.hLamina = ...
                smalltargetmotiondetectors.core.estmd_core.Lamina();
            self.hMedulla = ...
                smalltargetmotiondetectors.core.estmd_core.Medulla();
            self.hLobula = ...
                smalltargetmotiondetectors.core.estmd_core.Lobula();
        end
        
        function init_config(self)
            % INIT Method
            %   Initializes the ESTMD components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the retina, lamina, and medulla components.
            
            self.hRetina.init_config();
            self.hLamina.init_config();
            self.hMedulla.init_config();
        end

        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the ESTMD model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the ESTMD model
            %   components (retina, lamina, medulla, and lobula) and
            %   generates the model's response.

            % Process input matrix through model components
            self.retinaOpt = self.hRetina.process(iptMatrix);
            self.laminaOpt = self.hLamina.process(self.retinaOpt);
            self.hMedulla.process(self.laminaOpt);
            % self.lobulaOpt = self.hLobula.process(self.medullaOpt);
            self.medullaOpt = self.hMedulla.Opt;
            self.lobulaOpt = self.hLobula.process(self.medullaOpt);
            self.modelOpt.response = self.lobulaOpt;
        end

    end
end
