classdef STMDPlus < smalltargetmotiondetectors.model.DSTMDBackbone
    % STMDPlus - 
    %   This class implements an extended version of the Directional-
    %   Small Target Motion Detector (DSTMD) by inheriting from the 
    % DSTMDBackbone class.
    %
    % Ref: H. Wang, J. Peng, X. Zheng, S. Yue, A robust visual system for 
    % small target motion detection against cluttered moving backgrounds, 
    % IEEE Transactions on Neural Networks and Learning Systems 31 (3) 
    % (2020) 839–853.
    
    properties
        hContrastPathway; % Handle for the contrast pathway
        hMushroomBody; % Handle for the mushroom body
    end

    properties(Hidden)
        direContrastOpt; % Directional contrast pathway output
        mushroomBodyOpt; % Mushroom body output
    end
     
    
    methods
        function self = STMDPlus()
            % STMDPlus Constructor method
            %   Initializes an instance of the STMDPlus class.
            %
            % Syntax:
            %   obj = STMDPlus()
            %
            % Description:
            %   Initializes the STMDPlus object and sets up its components.
            
            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.DSTMDBackbone();
            % Import necessary packages
            import smalltargetmotiondetectors.core.stmdplus_core.*;

            % Initialize contrast pathway and mushroom body components
            self.hContrastPathway = ...
                smalltargetmotiondetectors.core.stmdplus_core.ContrastPathway();
            self.hMushroomBody = ...
                smalltargetmotiondetectors.core.stmdplus_core.MushroomBody();
        end
        
        function init(self)
            % INIT Method
            %   Initializes the STMDPlus components.
            %
            % Syntax:
            %   obj.init()
            %
            % Description:
            %   Initializes the components inherited from DSTMDBackbone
            %   as well as the contrast pathway and mushroom body.
            
            % Call superclass init method
            init@smalltargetmotiondetectors.model.DSTMDBackbone(self);

            % Initialize contrast pathway and mushroom body
            self.hContrastPathway.init();
            self.hMushroomBody.init();
        end

        function model_structure(self, iptMatrix)
            % MODEL_STRUCTURE Method
            %   Defines the structure of the STMDPlus model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the STMDPlus model
            %   components (retina, lamina, medulla, lobula, contrast pathway,
            %   and mushroom body) and generates the model's response.
            
            % Import necessary packages
            import smalltargetmotiondetectors.tool.compute.*;
            
            %% A. Ommatidia (Retina)
            self.retinaOpt = self.hRetina.process(iptMatrix);

            %% B. Motion Pathway (Lamina, Medulla, Lobula)
            self.laminaOpt = self.hLamina.process(self.retinaOpt);
            self.hMedulla.process(self.laminaOpt);
            self.medullaOpt = self.hMedulla.Opt;
            self.lobulaOpt = self.hLobula.process(self.medullaOpt);

            %% C. Contrast Pathway
            self.direContrastOpt = ...
                self.hContrastPathway.process(self.retinaOpt);

            %% D. Mushroom Body
            self.mushroomBodyOpt = self.hMushroomBody.process( ...
                self.lobulaOpt, ...
                self.direContrastOpt);

            %% Compute response and direction
            self.modelOpt.response = ...
                compute_response(self.mushroomBodyOpt);
            self.modelOpt.direction = ...
                compute_direction(self.mushroomBodyOpt);

        end

    end
end
