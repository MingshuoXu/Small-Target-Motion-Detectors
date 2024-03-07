function frackernel = create_fracdiff_kernel(alpha, wide)
    %CREATE_FRACDIFF_KERNEL Generates a fractional difference kernel.
    %   This function generates a fractional difference kernel based on the
    %   specified alpha and width.
    %
    %   Parameters:
    %   - alpha: The fractional difference parameter.
    %   - wide: The width of the kernel.
    %
    %   Returns:
    %   - frackernel: The fractional difference kernel.
    
    % Ensure the width is at least 2
    if wide < 2
        wide = 2;
    end
    
    % Initialize the kernel
    frackernel = zeros(1, wide);
    
    % Generate the kernel based on alpha
    if alpha == 1
        frackernel(1) = 1;
        frackernel(2:end) = 0;
    elseif 0 < alpha && alpha < 1
        tList = 0:wide-1;
        frackernel = exp(-alpha*tList/(1-alpha)) / (1-alpha);

        % Normalize the kernel
        sum_Kernel = sum(frackernel); % 1/M(\alpha)
        frackernel = frackernel / sum_Kernel;
    else
        error('Alpha must be in the interval (0,1]. \n');
    end
end
