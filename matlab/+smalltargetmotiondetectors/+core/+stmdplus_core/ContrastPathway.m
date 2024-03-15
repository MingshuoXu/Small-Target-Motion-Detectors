classdef ContrastPathway < smalltargetmotiondetectors.core.BaseCore
    % ContrastPathway class for ApgSTMD

    properties
        theta = [0:3]*pi/4; % Angles for filters
        alpha2 = 1.5; % Alpha parameter for kernel creation
        eta = 3; % Eta parameter for kernel creation
        sizeT1 = 11; % Size parameter for kernel creation
    end

    properties(Hidden)
        T1Kernel; % T1 kernels
    end

    methods
        function self = ContrastPathway()
            % Constructor method
            % Initializes the ContrastPathway object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes the T1Kernel

            import smalltargetmotiondetectors.tool.kernel.*;

            self.T1Kernel = create_T1_kernels(...
                length(self.theta), ...
                self.alpha2, ...
                self.eta, ...
                self.sizeT1 ...
                );
        end

        function dictContrastOpt = process(self, retinaOpt)
            % Processing method
            % Processes the input retinaOpt to generate contrastOpt

            lenKernel = length(self.theta);
            dictContrastOpt = zeros([size(retinaOpt), lenKernel]);
            for idx = 1:lenKernel
                dictContrastOpt(:,:,idx) = ...
                    conv2(retinaOpt , self.T1Kernel{idx}, 'same');
            end
            self.Opt = dictContrastOpt;
        end
    end

end
