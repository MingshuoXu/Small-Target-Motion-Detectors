classdef GaussianBlur < smalltargetmotiondetectors.core.BaseCore
    % GAUSSIANBLUR Gaussian blur filter
    %   This class implements a Gaussian blur filter for image processing.
    %
    
    properties
        size = 3;   % Size of the filter kernel
        sigma = 1;  % Standard deviation of the Gaussian distribution
    end
    
    properties (Hidden)
        gaussKernel;    % Gaussian filter kernel
    end
    
    methods
        function self = GaussianBlur()
            % Constructor
            % Initializes the GaussianBlur object
        end
    end
    
    methods
        function init_config(self)
            % Initialization method
            % Creates the Gaussian filter kernel using fspecial function
            import smalltargetmotiondetectors.util.kernel.*;
            try
                self.gaussKernel = ...
                    fspecial('gaussian', self.size, self.sigma);
            catch
                self.gaussKernel = ...
                    create_gaussian_kernel(self.size, self.sigma);
            end
        end

        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Applies the Gaussian filter to the input matrix
            %   iptMatrix: input matrix
            %   optMatrix: output matrix
            
            optMatrix = conv2(iptMatrix, self.gaussKernel, 'same');
        end
    end
end
