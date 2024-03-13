classdef LobulaPlateTangentialCell < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs several operations including temporal
    %   convolution, correlation, and surround inhibition.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        hSubInhi;  % SurroundInhibition component
        alpha = 1;  % Parameter alpha
        % Parameters for gamma kernel
        paraGammaKernel = struct('order', 10, 'tau', 25, 'len', []);  
        % Parameters for Gaussian kernel
        paraGaussKernel = struct('eta', 1.5, 'size', 3);  
    end
    properties(Hidden)
        gammaKernel;  % Gamma kernel
        gaussKernel;  % Gaussian kernel
        hGammaDelay;
    end


    methods
        function self = LobulaPlateTangentialCell()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.*;
            
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();
            self.hGammaDelay = GammaDelay(10, 25);
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            import smalltargetmotiondetectors.tool.kernel.*;
            
            % Initialize parameters and kernels
            self.hSubInhi.init();

            self.hGammaDelay.init();

            self.gaussKernel = fspecial(...
                'gaussian', ...
                self.paraGaussKernel.size, ...
                self.paraGaussKernel.eta);
        end

        function lateralInhiSTMDopt = process(self, varagein)
            % Processing method
            % Performs temporal convolution, correlation, and surround
            % inhibition
            
            import smalltargetmotiondetectors.tool.compute.*;
            
            % Extract ON and OFF channel signals from the input
            onSignal = varagein{1};
            offSignal = varagein{2};

            % Formula (11)
            feedbackSignal = self.alpha * ...
                self.hGammaDelay.process( ...
                zeros(size(onSignal)) ...
                );
            
            %Fai and theta


            % Formula (10)
            % There must have the max operation here
            correlationD = ...
                max( (onSignal - feedbackSignal), 0) .* ...
                max( (offSignal - feedbackSignal), 0);
            
            % Formula (13)
            correlationE = ...
                conv2(onSignal.*offSignal, self.gaussKernel, 'same');

            % Formula (14)
            % lateralInhiSTMDopt: STMD output after lateral inhibition in formula (14)
            lateralInhiSTMDopt = self.hSubInhi.process(correlationD);

            % only record correlationD + correlationE
            self.hGammaDelay.isCircshift = false;
            self.hGammaDelay.process( correlationD + correlationE );
            self.hGammaDelay.isCircshift = true;


            % Store the output in Opt property
            self.Opt = lateralInhiSTMDopt;
        end


    end

end
