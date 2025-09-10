classdef SurroundInhibition < smalltargetmotiondetectors.core.BaseCore
    %Gamma_Filter Gamma filter in lamina layer
    %   References:
    %   * S. D. Wiederman, P. A. Shoemarker, D. C. O¡¯Carroll, A model
    %   for the detection of moving targets in visual clutter inspired by
    %   insect physiology, PLoS ONE 3 (7) (2008) e2784¨C.
    %           Page5: In addition to this step, a second subtractive 
    %       inhibition is applied based on the average of the surrounding 
    %       input signals of the same channel polarity (surrounding ON 
    %       subtractively inhibit the centre ON channel and similarly for 
    %       the OFF channels). This is based on the surround inhibitory 
    %       effect found in the on-off cells [21].
    %
    %   * Wang H, Peng J, Yue S. A directionally selective small target
    %   motion detecting visual neural network in cluttered backgrounds[J].
    %   IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    %   LastEditTime: 2024-02-22

    properties
        KernelSize = 15;    % Size of the filter kernel
        Sigma1 = 1.5;       % Standard deviation for the first Gaussian
        Sigma2 = 3;         % Standard deviation for the second Gaussian
        e = 1.0;            % Exponent for the weighting of the second Gaussian
        rho = 0;            % Radius for circular integration
        A = 1;              % Amplitude of the filter
        B = 3;              % Offset of the filter
    end
    
    properties (Hidden)
        inhiKernelW2;       % Surround inhibition filter kernel
    end
    
    methods
        function self = SurroundInhibition(KernelSize, Sigma1, Sigma2, e, rho, A, B)
            % Constructor
            % Initializes the SurroundInhibition object with optional parameters
            
            if nargin >= 1
                self.KernelSize = KernelSize;
            end
            if nargin >= 2
                self.Sigma1 = Sigma1;
            end
            if nargin >= 3
                self.Sigma2 = Sigma2;
            end
            if nargin >= 4
                self.e = e;
            end
            if nargin >= 5
                self.rho = rho;
            end
            if nargin >= 6
                self.A = A;
            end
            if nargin >= 7
                self.B = B;
            end
        end
        
        function init_config(self)
            % Initialization method
            % Creates the surround inhibition filter kernel
            
            import smalltargetmotiondetectors.util.kernel.*;
            
            self.inhiKernelW2 = create_inhi_kernel_W2(...
                self.KernelSize, ...
                self.Sigma1, ...
                self.Sigma2, ...
                self.e, ...
                self.rho, ...
                self.A, ...
                self.B);
        end
        
        function inhiOpt = process(self, iptMatrix)
            % Processing method
            % Applies the surround inhibition filter to the input matrix
            
            inhiOpt = conv2(iptMatrix, self.inhiKernelW2, 'same');
            inhiOpt = max(inhiOpt, 0);
        end
    end
end



