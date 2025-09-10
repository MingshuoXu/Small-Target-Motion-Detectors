classdef DirectionInhi < smalltargetmotiondetectors.core.BaseCore
    % Directional inhibition in DSTMD
    %   This class represents the directional inhibition component in the
    %   motion detection system. It performs inhibition based on the
    %   direction of motion.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        direction = 8;  % Number of directions
        sigma1 = 1.5;  % Sigma for the first Gaussian kernel
        sigma2 = 3.0;  % Sigma for the second Gaussian kernel
    end

    properties(Hidden)
        diretionalInhiKernel;  % Directional inhibition kernel
    end

    methods
        function self = DirectionInhi()
            % Constructor method
            % Initializes the DirectionInhi object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
        end

        function init_config(self)
            % Initialization method
            % This method initializes the directional inhibition kernel
            
            import smalltargetmotiondetectors.util.kernel.*;
            
            % Create directional inhibition kernel
            if isempty(self.diretionalInhiKernel)
                self.diretionalInhiKernel = ...
                    create_direction_inhi_kernel(...
                    self.direction, ...
                    self.sigma1, ...
                    self.sigma2 ...
                    );
            end
            self.diretionalInhiKernel = squeeze(self.diretionalInhiKernel);
        end

        function opt = process(self, iptCell)
            % Processing method
            % Performs directional inhibition on the input
            
            len1 = length(iptCell);
            len2 = length(self.diretionalInhiKernel);

            certer = ceil(len2/2);
            RR = certer-1;

            if mod(len2,2) ~= 0
                LL = - RR;
            else
                LL = - RR + 1;
            end

            opt = cell(len1, 1);
            [m, n] = size(iptCell{1});
 
            for idx = 1:len1
                opt{idx} = zeros(m, n);
                for j = LL : RR
                    k = mod(idx-j, len1);
                    if k == 0
                        k = len1;
                    end
                    opt{idx} = opt{idx} ...
                        + iptCell{k} * self.diretionalInhiKernel(j+certer);
                end
                opt{idx} = max(opt{idx}, 0);
            end

        end % [EoF]
    end
end
