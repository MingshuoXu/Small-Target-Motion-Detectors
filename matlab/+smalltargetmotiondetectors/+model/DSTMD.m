classdef DSTMD < smalltargetmotiondetectors.model.BaseModel
    % DSTMD - Directional-Small Target Motion Detector
    %   This class implements a Directional-Small Target Motion Detector
    %   by inheriting from the BaseModel class.
    %
    % Ref: H. Wang, J. Peng, S. Yue, A directionally selective small target
    % motion detecting visual neural network in cluttered backgrounds, IEEE
    % Transactions on Cybernetics 50 (4) (2020) 1541–1555.
    
    properties
        % Define properties here (if any)
    end
     

    
    methods
        function self = DSTMD()
            % DSTMD Constructor method
            %   Initializes an instance of the DSTMD class.
            %
            % Syntax:
            %   obj = DSTMD()
            %
            % Description:
            %   Initializes the DSTMD object and sets up its components.

            % Call superclass constructor
            self = self@smalltargetmotiondetectors.model.BaseModel();

            % Import necessary packages
            import smalltargetmotiondetectors.core.estmd_core.*;
            import smalltargetmotiondetectors.core.dstmd_core.*;


            % Initialize components
            self.hRetina = ...
                smalltargetmotiondetectors.core.estmd_core.Retina();
            self.hLamina = ...
                smalltargetmotiondetectors.core.estmd_core.Lamina();
            self.hMedulla = ...
                smalltargetmotiondetectors.core.dstmd_core.Medulla();
            self.hLobula = ...
                smalltargetmotiondetectors.core.dstmd_core.Lobula();
        end
        
        function init(self)
            % INIT Method
            %   Initializes the DSTMD components.
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
            %   Defines the structure of the DSTMD model.
            %
            % Syntax:
            %   obj.model_structure(iptMatrix)
            %
            % Input:
            %   iptMatrix - Input matrix for processing
            %
            % Description:
            %   Processes the input matrix through the DSTMD model
            %   components (retina, lamina, medulla, and lobula) and
            %   generates the model's response.
            
            % Import necessary packages
            import smalltargetmotiondetectors.tool.compute.*;
            
            % Process input matrix through model components
            self.retinaOpt = self.hRetina.process(iptMatrix);
            self.laminaOpt = self.hLamina.process(self.retinaOpt);
            self.hMedulla.process(self.laminaOpt);
            self.medullaOpt = self.hMedulla.Opt;
            self.lobulaOpt = self.hLobula.process(self.medullaOpt);

            % Compute response and direction
            self.modelOpt.response = compute_response(self.lobulaOpt);
            self.modelOpt.direction = compute_direction(self.lobulaOpt);
        end

    end
end
