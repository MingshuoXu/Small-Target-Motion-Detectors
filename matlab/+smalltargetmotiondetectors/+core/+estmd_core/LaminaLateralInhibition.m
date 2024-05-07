classdef LaminaLateralInhibition < smalltargetmotiondetectors.core.BaseCore
    % LAMINALATERALINHIBITION Lateral inhibition in the Lamina layer
    %   This class implements the lateral inhibition mechanism in the Lamina layer
    %   of the ESTMD.
    %
    %   References:
    %   * S. D. Wiederman, P. A. Shoemarker, D. C. O'Carroll, A model
    %   for the detection of moving targets in visual clutter inspired by
    %   insect physiology, PLoS ONE 3 (7) (2008) e2784�C.
    %   * Wang H, Peng J, Yue S. A directionally selective small target
    %   motion detecting visual neural network in cluttered backgrounds[J].
    %   IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    %   LastEditTime: 2023-02-22
    
    properties
        sizeW1 = [11, 11, 7];  % Size of the inhibition kernel W1
        sigma2 = 1.5;      % Standard deviation for the positive part of W1
        sigma3;            % Standard deviation for the negative part of W1
        lambda1 = 3;       % Time constant for the positive part of W1
        lambda2 = 9;       % Time constant for the negative part of W1
    end
    
    properties(Hidden)
        spatialPositiveKernel;          % Positive part of the inhibition kernel W1
        spatialNegativeKernel;          % Negative part of the inhibition kernel W1
        temporalPositiveKernel;          % Temporal component for the positive part of W1
        temporalNegativeKernel;          % Temporal component for the negative part of W1
        % Cell array to store intermediate results for the positive part
        cellSpatialPositive;      
        % Cell array to store intermediate results for the negative part
        cellSpatialNegative;          
    end
    
    methods
        function self = LaminaLateralInhibition(...
                sizeW1, lambda1, lambda2, sigma2, sigma3)
            % Constructor
            % Initializes the LaminaLateralInhibition object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            if nargin >= 1
                self.sizeW1 = sizeW1;
            end
            if nargin >= 2
                self.lambda1 = lambda1;
            end
            if nargin >= 3
                self.lambda2 = lambda2;
            end
            if nargin >= 4
                self.sigma2 = sigma2;
            end
            if nargin >= 5
                self.sigma3 = sigma3;
            end
        end
        
        function init_config(self)
            % Initialization method
            % Initializes the inhibition kernel W1
            import smalltargetmotiondetectors.util.*;
            import smalltargetmotiondetectors.util.compute.*;
            import smalltargetmotiondetectors.util.kernel.*;
            
            if isempty(self.sigma3)
                self.sigma3 = 2 * self.sigma2;
            end
            
            try
                gaussionSigma2 = ...
                    fspecial('gaussian', self.sizeW1(1:2), self.sigma2);
            catch
                gaussionSigma2 = ...
                    create_gaussian_kernel(self.sizeW1(1:2), self.sigma2);
            end
            
            try
                gaussionSigma3 = ...
                    fspecial('gaussian', self.sizeW1(1:2), self.sigma3);
            catch
                gaussionSigma3 = ...
                    create_gaussian_kernel(self.sizeW1(1:2), self.sigma3);
            end
            
            DiffOfGaussion = gaussionSigma2 - gaussionSigma3;
            % W_{S}^{P} in formulate (8) of DSTMD
            self.spatialPositiveKernel = max(DiffOfGaussion, 0);
            % W_{S}^{N} in formulate (9) of DSTMD
            self.spatialNegativeKernel = max(-DiffOfGaussion, 0);
            
            t = (1:self.sizeW1(3)) - 1;
            % W_{T}^{P} in formulate (10) of DSTMD
            self.temporalPositiveKernel = exp(-t / self.lambda1) / self.lambda1;
            % W_{T}^{N} in formulate (11) of DSTMD
            self.temporalNegativeKernel = exp(-t / self.lambda2) / self.lambda2;
            
            self.cellSpatialPositive = CircularCell(self.sizeW1(3));
            self.cellSpatialNegative = CircularCell(self.sizeW1(3));
        end
        
        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Applies lateral inhibition to the input matrix
            
            import smalltargetmotiondetectors.util.compute.compute_circularcell_conv;

            % Lateral inhibition
            self.cellSpatialPositive.circrecord( ...
                conv2(iptMatrix, self.spatialPositiveKernel, 'same') ...
                );

            self.cellSpatialNegative.circrecord( ...
                conv2(iptMatrix, self.spatialNegativeKernel, 'same')...
                );

            optMatrix ...
                = compute_circularcell_conv( ...
                self.cellSpatialPositive, ...
                self.temporalPositiveKernel) ...
                + ...
                compute_circularcell_conv( ...
                self.cellSpatialNegative, ...
                self.temporalNegativeKernel);

            if isempty(optMatrix)
                optMatrix = zeros(size(iptMatrix));
            end
        end
    end
    
end
