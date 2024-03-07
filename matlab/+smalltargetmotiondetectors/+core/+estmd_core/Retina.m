classdef Retina < smalltargetmotiondetectors.core.BaseCore
    % RETINA Retina filter
    %   This class implements Retina Layer.
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    
    properties
        hGaussianBlur;  % Handle to the GaussianBlur object
    end
    
    methods
        function self = Retina()
            % Constructor
            % Initializes the Retina object and creates a GaussianBlur object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.GaussianBlur;
            
            self.hGaussianBlur = GaussianBlur();
        end
    end
    
    methods
        function init(self)
            % Initialization method
            % Initializes the GaussianBlur object
            
            self.hGaussianBlur.init();
        end
        
        function retinaOpt = process(self, retinaIpt)
            % Processing method
            % Applies the Gaussian blur filter to the input matrix
            
            retinaOpt = self.hGaussianBlur.process(retinaIpt);
            self.Opt = retinaOpt;
        end
    end
end
