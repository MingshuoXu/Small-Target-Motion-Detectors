function inhibitionKernelW2 = create_inhi_kernel_W2( ...
        KernelSize, ...
        Sigma1, ...
        Sigma2, ...
        e, ...
        rho, ...
        A, ...
        B)
    % CREATE_INHI_KERNEL_W2 Generates the lateral inhibition convolution 
    % kernel for DSTMD.
    %   This function generates the lateral inhibition convolution kernel 
    %   for the Directional Small Target Motion Detector (DSTMD),
    %   in the form of Difference of Gaussians (DoG).
    %
    %   Formula:
    %   W(x,y) = A*[g_1(x,y)] - B[-g_1(x,y)]    % [x] max(x,0)
    %   g_1 = G_1(x,y) - e*G_2(x,y) - rho
    %
    %   Parameters:
    %   - KernelSize: Size of the inhibition kernel. Typically an odd number.
    %   - Sigma1: Sigma of the Gaussian function 1.
    %   - Sigma2: Sigma of the Gaussian function 2.
    %   - e: Parameter e.
    %   - rho: Parameter rho.
    %   - A: Coefficient A.
    %   - B: Coefficient B.
    %
    %   Returns:
    %   - inhibitionKernelW2: The generated inhibition kernel.
    %
    %   Author: Hongxin Wang
    %   Date: 2021-05-13
    %   LastEdit: Mingshuo Xu
    %   LastEditTime: 2022-07-11

    %% ----------------------------------------------%
    
    % Set default values
    if nargin < 1
        KernelSize = 15;
    end
    if nargin < 2
        Sigma1 = 1.5;
    end
    if nargin < 3
        Sigma2 = 3;
    end
    if nargin < 4
        e = 1.0;
    end
    if nargin < 5
        rho = 0;
    end
    if nargin < 6
        A = 1;
    end
    if nargin < 7
        B = 3;
    end

    % Ensure KernelSize is odd
    if mod(KernelSize,2) == 0
        KernelSize = KernelSize + 1;
    end
    
    % Determine the center of the kernel
    CenX = round(KernelSize/2);
    CenY = round(KernelSize/2);
    
    % Generate grid
    [ShiftX,ShiftY] = ...
        meshgrid((1:KernelSize) - CenX, (KernelSize:-1:1) - CenY);
        
    % Generate Gauss functions 1 and 2
    Gauss1 = (1 / (2 * pi * Sigma1^2)) * ...
        exp(-(ShiftX.^2 + ShiftY.^2) / (2 * Sigma1^2));
    Gauss2 = (1 / (2 * pi * Sigma2^2)) * ...
        exp(-(ShiftX.^2 + ShiftY.^2) / (2 * Sigma2^2));
    
    % Generate DoG, subtracting two Gaussian functions
    DoG_Filter = Gauss1 - e * Gauss2 - rho;
    
    % max(x,0)
    Positive_Component = max(DoG_Filter, 0);
    Negative_Component = max(-DoG_Filter, 0);
    
    % Inhibition Kernel
    inhibitionKernelW2 = A * Positive_Component - B * Negative_Component;
end
