classdef ESTMDBackbone < smalltargetmotiondetectors.model.BaseModel
    % ESTMDBackbone - Backbone model based on ESTMD
    %   
    %
    % Ref: S.D. Wiederman, P.A. Shoemarker, D.C. O Carroll, A model for the
    % detection of moving targets in visual clutter inspired by insect 
    % physiology, PLoS ONE 3 (7) (2008) e2784.
    % (the first STMD-based model)
    
    properties
        % Define properties here (if any)
    end
     

    
    methods
        function self = ESTMDBackbone()
            % ESTMDBackbone Constructor method
            %   Initializes an instance of the ESTMDBackbone class.
            %
            % Syntax:
            %   obj = ESTMDBackbone()
            %
            % Description:
            %   Initializes the ESTMDBackbone object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.BaseModel();
            % Import necessary packages
            import smalltargetmotiondetectors.core.estmd_core.Retina;
            import smalltargetmotiondetectors.core.estmd_backbone.*;

            % Initialize components
            self.hRetina = smalltargetmotiondetectors.core.estmd_core.Retina();
            self.hLamina = smalltargetmotiondetectors.core.estmd_backbone.Lamina();
            self.hMedulla = smalltargetmotiondetectors.core.estmd_backbone.Medulla();
            self.hLobula = smalltargetmotiondetectors.core.estmd_backbone.Lobula();
        end
        
        function init(self)
            % INIT Method
            %   Initializes the ESTMDBackbone components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the retina, lamina, medulla, and lobula components.
            
            self.hRetina.init();
            self.hLamina.init();
            self.hMedulla.init();
            self.hLobula.init();
        end

        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the ESTMDBackbone model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the ESTMDBackbone model
            %   components (retina, lamina, medulla, and lobula) and
            %   generates the model's response.
            
            % Process input matrix through model components
            self.retinaOpt = self.hRetina.process(iptMatrix);
            self.laminaOpt = self.hLamina.process(self.retinaOpt);
            self.hMedulla.process(self.laminaOpt);
            % self.lobulaOpt = self.hLobula.process(self.medullaOpt);
            self.medullaOpt = self.hMedulla.Opt;
            self.lobulaOpt = self.hLobula.process(self.medullaOpt);

            % Set model response
            self.modelOpt.response = self.lobulaOpt;
        end

    end
end
