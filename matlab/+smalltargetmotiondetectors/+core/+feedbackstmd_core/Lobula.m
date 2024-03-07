classdef Lobula < smalltargetmotiondetectors.core.BaseCore
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
        cellDplusE;  % Cell array for storing intermediate results
        gammaKernel;  % Gamma kernel
        gaussKernel;  % Gaussian kernel
    end


    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.SurroundInhibition;
            
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            import smalltargetmotiondetectors.tool.kernel.*;
            
            % Initialize parameters and kernels
            if isempty(self.paraGammaKernel.len)
                self.paraGammaKernel.len = ...
                    3 * ceil(self.paraGammaKernel.tau);
            end

            self.gammaKernel = create_gamma_kernel( ...
                self.paraGammaKernel.order, ...
                self.paraGammaKernel.tau, ...
                self.paraGammaKernel.len);

            self.hSubInhi.init();
            self.cellDplusE = cell(self.paraGammaKernel.len, 1);
            self.gaussKernel = fspecial(...
                'gaussian', ...
                self.paraGaussKernel.size, ...
                self.paraGaussKernel.eta);
        end

        function lobulaOpt = process(self, varagein)
            % Processing method
            % Performs temporal convolution, correlation, and surround
            % inhibition
            
            import smalltargetmotiondetectors.tool.compute.*;
            
            % Extract ON and OFF channel signals from the input
            onSignal = varagein{1};
            offSignal = varagein{2};

            % Formula (9)
            self.cellDplusE(1) = [];
            self.cellDplusE{end+1} = zeros(size(onSignal));
            feedbackSignal = self.alpha * ...
                compute_temporal_conv(self.cellDplusE, self.gammaKernel);

            % Formula (8)
            correlationD = ...
                (onSignal - feedbackSignal) ...
                .* (offSignal - feedbackSignal);
            
            % Formula (10)
            correlationE = ...
                conv2(onSignal.*offSignal, self.gaussKernel, 'same');

            self.cellDplusE{end} = correlationD + correlationE;

            % Formula (14)
            lobulaOpt = self.hSubInhi.process(correlationD);

            % Store the output in Opt property
            self.Opt = lobulaOpt;
        end


    end

end
