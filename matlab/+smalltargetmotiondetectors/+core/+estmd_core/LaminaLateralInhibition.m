classdef LaminaLateralInhibition < smalltargetmotiondetectors.core.BaseCore
    % LAMINALATERALINHIBITION Lateral inhibition in the Lamina layer
    %   This class implements the lateral inhibition mechanism in the Lamina layer
    %   of the ESTMD.
    %
    %   References:
    %   * S. D. Wiederman, P. A. Shoemarker, D. C. O'Carroll, A model
    %   for the detection of moving targets in visual clutter inspired by
    %   insect physiology, PLoS ONE 3 (7) (2008) e2784¨C.
    %   * Wang H, Peng J, Yue S. A directionally selective small target
    %   motion detecting visual neural network in cluttered backgrounds[J].
    %   IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    %   LastEditTime: 2023-02-22
    
    properties
        size_W1 = [15, 15, 7];  % Size of the inhibition kernel W1
        sigma2 = 1.5;      % Standard deviation for the positive part of W1
        sigma3;            % Standard deviation for the negative part of W1
        lambda1 = 3;       % Time constant for the positive part of W1
        lambda2 = 9;       % Time constant for the negative part of W1
    end
    
    properties(Hidden)
        W_S_P;          % Positive part of the inhibition kernel W1
        W_S_N;          % Negative part of the inhibition kernel W1
        W_T_P;          % Temporal component for the positive part of W1
        W_T_N;          % Temporal component for the negative part of W1
        % Cell array to store intermediate results for the positive part
        cell_BP_W_S_P;      
        % Cell array to store intermediate results for the negative part
        cell_BP_W_S_N;          
    end
    
    methods
        function self = LaminaLateralInhibition(...
                size_W1, lambda1, lambda2, sigma2, sigma3)
            % Constructor
            % Initializes the LaminaLateralInhibition object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            if nargin >= 1
                self.size_W1 = size_W1;
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
        
        function init(self)
            % Initialization method
            % Initializes the inhibition kernel W1
            
            if isempty(self.sigma3)
                self.sigma3 = 2 * self.sigma2;
            end

            G_sigma2 = fspecial('gaussian', self.size_W1(1:2), self.sigma2);
            G_sigma3 = fspecial('gaussian', self.size_W1(1:2), self.sigma3);
            temp = G_sigma2 - G_sigma3;
            self.W_S_P = max(temp, 0);
            self.W_S_N = max(-temp, 0);
            
            t = (1:self.size_W1(3)) - 1;
            self.W_T_P = exp(-t / self.lambda1) / self.lambda1;
            self.W_T_N = exp(-t / self.lambda2) / self.lambda2;
            
            self.cell_BP_W_S_P = cell(self.size_W1(3), 1);
            self.cell_BP_W_S_N = cell(self.size_W1(3), 1);
        end
        
        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Applies lateral inhibition to the input matrix
            
            import smalltargetmotiondetectors.tool.compute.*;

            % Lateral inhibition
            self.cell_BP_W_S_P = circshift(self.cell_BP_W_S_P, -1);
            self.cell_BP_W_S_P{end} = conv2(iptMatrix, self.W_S_P, 'same');
            self.cell_BP_W_S_N = circshift(self.cell_BP_W_S_N, -1);
            self.cell_BP_W_S_N{end} = conv2(iptMatrix, self.W_S_N, 'same');

            optMatrix = compute_temporal_conv( ...
                self.cell_BP_W_S_P, self.W_T_P) + ...
                compute_temporal_conv( ...
                self.cell_BP_W_S_N, self.W_T_N);

            if isempty(optMatrix)
                optMatrix = zeros(size(iptMatrix));
            end
        end
    end
    
end
