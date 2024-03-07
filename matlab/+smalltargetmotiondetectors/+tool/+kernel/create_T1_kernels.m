function dictKernel = create_T1_kernels(numfilter, alpha, eta, filterSize)
    % CREATE_T1_KERNELS Generates convolution kernels for STMDPlus T1 neurons.
    %   This function generates convolution kernels for STMDPlus T1 neurons,
    %   which are composed of filters defined as:
    %   Filter = G(x - a*cos(theta), y - a*sin(theta)) - 
    %            G(x + a*cos(theta), y + a*sin(theta))
    %
    %   Parameters:
    %   - numfilter: Number of filters.
    %   - alpha: Distance between the center of the Gaussian function and the filter center.
    %   - eta: Sigma of the Gaussian function.
    %   - filterSize: Size of the filter.
    %
    %   Returns:
    %   - dictKernel: Cell array containing the generated convolution kernels.
    %
    %   Ref: Construction and Evaluation of an Integrated Dynamical      
    %   Model of Visual Motion Perception

    % Default parameter values
    if nargin < 1
        numfilter = 4;
    end
    if nargin < 2
        filterSize = 11;
    elseif mod(filterSize, 2) == 0
        filterSize = filterSize + 1;
        % If the filter size is even, force it to be odd
    end
    if nargin < 3
        alpha = 3.0;
    end
    if nargin < 4
        eta = 1.5;
    end

    % Compute angles for each filter
    Theta = zeros(1, numfilter);
    for i = 1:numfilter
        Theta(i) = (i - 1) * pi / numfilter;
    end

    % Initialize cell array to store generated kernels
    dictKernel = cell(numfilter, 1);

    % Generate coordinates
    r = floor(filterSize / 2);
    [X, Y] = meshgrid(-r:r, -r:r);

    % Generate kernels for each filter
    for idx = 1:numfilter
        % Determine the centers of the two Gaussian functions
        X1 = X - alpha * cos(Theta(idx));
        Y1 = Y - alpha * sin(Theta(idx));
        X2 = X + alpha * cos(Theta(idx));
        Y2 = Y + alpha * sin(Theta(idx));

        % Generate the two Gaussian functions
        Gauss1 = (1 / (2 * pi * eta^2)) * exp(-(X1.^2 + Y1.^2) / (2 * eta^2));
        Gauss2 = (1 / (2 * pi * eta^2)) * exp(-(X2.^2 + Y2.^2) / (2 * eta^2));

        % Create the filter as the difference between the two Gaussians
        dictKernel{idx} = Gauss1 - Gauss2;
    end
end
