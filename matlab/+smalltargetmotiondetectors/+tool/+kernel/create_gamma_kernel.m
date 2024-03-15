function gammaKernel = create_gamma_kernel(Order, Tau, wide)
    % CREATE_GAMMA_KERNEL Generates a discretized Gamma vector.
    %   This function generates a discretized Gamma vector with a specified
    %   order, time constant (Tau), and length (wide). The independent variable
    %   t ranges from 0 to wide-1.
    %
    %   Parameters:
    %   - Order: The order of the Gamma function.
    %   - Tau: The time constant of the Gamma function.
    %   - wide: The length of the vector.
    %
    %   Returns:
    %   - gammaKernel: The generated Gamma vector.

    if nargin < 1
        Order = 100;
    end
    if nargin < 2
        Tau = 25;
    end
    if nargin < 3
        wide = ceil(3*Tau);
    end

    % Ensure wide is at least 2
    if wide <= 1
        wide = 2;
    end

    % Initialize the Gamma vector
    gammaKernel = zeros(1, wide);

    % Compute the values of the Gamma vector
    timeList = 0:wide-1;
    gammaKernel = ...
        (Order * timeList / Tau).^Order .* exp(-Order * timeList / Tau) ...
        ./ (factorial(Order - 1) * Tau);

    % Normalize the Gamma vector
    gammaKernel = gammaKernel / sum(gammaKernel, 'all');
    gammaKernel(gammaKernel<1e-4) = 0;
    gammaKernel = gammaKernel / sum(gammaKernel, 'all');

end
