classdef GammaDelay < smalltargetmotiondetectors.core.BaseCore
    % GAMMADELAY Gamma filter in lamina layer
    %   References:
    %   * S. D. Wiederman, P. A. Shoemarker, D. C. O'Carroll, A model
    %     for the detection of moving targets in visual clutter inspired by
    %     insect physiology, PLoS ONE 3 (7) (2008) e2784.
    %           Page 5: In addition to this step, a second subtractive 
    %           inhibition is applied based on the average of the surrounding 
    %           input signals of the same channel polarity (surrounding ON 
    %           subtractively inhibit the centre ON channel and similarly for 
    %           the OFF channels). This is based on the surround inhibitory 
    %           effect found in the 'on-off' cells [21].
    %
    %   * Wang H, Peng J, Yue S. A directionally selective small target
    %     motion detecting visual neural network in cluttered backgrounds[J].
    %     IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    %   LastEditTime: 2024-02-22

    properties
        order;      % Order of the gamma filter
        tau;        % Time constant of the filter
        lenKernal;  % Length of the filter kernel
        
        isCircshift = true; % Flag to indicate circular shifting
    end

    
    properties (Hidden)
        gammaKernel;    % Gamma filter kernel
        cellInputHist;  % Input history stored in a cell array
    end
    
    methods
        function self = GammaDelay(order, tau, lenKernal)
            % Constructor
            % Initializes the GammaDelay object with provided parameters
            self = self@smalltargetmotiondetectors.core.BaseCore();
            
            if nargin >= 1
                self.order = order;
            end
            if nargin >= 2
                self.tau = tau;
            end
            if nargin >= 3
                self.lenKernal = lenKernal;
            end
        end

        function init(self)
            % Initialization method
            % Creates the gamma filter kernel and initializes input history
            import smalltargetmotiondetectors.tool.kernel.*;

            if isempty(self.lenKernal)
                self.lenKernal = 3 * ceil(self.tau);
            end

            if isempty(self.gammaKernel)
                self.gammaKernel = create_gamma_kernel( ...
                    self.order, ...
                    self.tau, ...
                    self.lenKernal);
            end

            self.cellInputHist = cell(self.lenKernal, 1);
        end

        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Applies the gamma filter to the input matrix
            
            import smalltargetmotiondetectors.tool.compute.*;

            if self.isCircshift
                self.cellInputHist = circshift(self.cellInputHist, -1);
            end
            self.cellInputHist{end} = iptMatrix;
            
            optMatrix = compute_temporal_conv(...
                self.cellInputHist, ...
                self.gammaKernel);
        end
    end
end
