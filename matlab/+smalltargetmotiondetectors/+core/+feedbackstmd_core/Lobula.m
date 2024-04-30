classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs several operations including temporal
    %   convolution, correlation, and surround inhibition.
    %
    %   Date: 2024-03-10

    properties
        hSubInhi;  % SurroundInhibition component
        alpha = 1;  % Parameter alpha 
        % Parameters for Gaussian kernel
        paraGaussKernel = struct('eta', 1.5, 'size', 3);  
    end
    properties(Hidden)
        gammaKernel;  % Gamma kernel
        gaussKernel;  % Gaussian kernel
        hGammaDelay;
    end


    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.SurroundInhibition;
            import smalltargetmotiondetectors.core.math_operator.GammaDelay;
            
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();
            self.hGammaDelay = GammaDelay(10, 25);
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            import smalltargetmotiondetectors.util.kernel.*;
            
            % Initialize parameters and kernels
            self.hSubInhi.init_config();

            self.hGammaDelay.init_config();

            self.gaussKernel = fspecial(...
                'gaussian', ...
                self.paraGaussKernel.size, ...
                self.paraGaussKernel.eta);
        end

        function lobulaOpt = process(self, varargin)
            % Processing method
            % Performs temporal convolution, correlation, and surround
            % inhibition
            
            import smalltargetmotiondetectors.util.compute.*;
            
            % Extract ON and OFF channel signals from the input
            onSignal = varargin{1};
            offSignal = varargin{2};

            % Formula (9)
            feedbackSignal = self.alpha * ...
                self.hGammaDelay.process( zeros(size(onSignal)) );
            
            % Formula (8)
            % There must have the max operation here
            correlationD = ...
                max( (onSignal - feedbackSignal), 0) .* ...
                max( (offSignal - feedbackSignal), 0);
            
            % Formula (10)
            correlationE = ...
                conv2(onSignal.*offSignal, self.gaussKernel, 'same');

            % only record correlationD + correlationE
            self.hGammaDelay.hCellInput.cover(correlationD + correlationE);
            

            % Formula (14)
            lobulaOpt = self.hSubInhi.process(correlationD);

            % Store the output in Opt property
            self.Opt = lobulaOpt;
        end


    end

end
