function optMatrix = compute_temporal_conv(iptCell, kernel, headPointer)
    % COMPUTE_TEMPORAL_CONV Computes temporal convolution.
    %   Computes temporal convolution of the input cell array with the given
    %   kernel, starting from the head pointer position.
    %
    %   Parameters:
    %   - iptCell: A cell array where each element has the same dimension.
    %   - kernel: A vector representing the convolution kernel.
    %   - headPointer: Head pointer of the input cell array.
    %
    %   Returns:
    %   - optMatrix: The result of the temporal convolution.

    % Default value for headPointer
    if ~exist('headPointer','var')
        headPointer = length(iptCell);
    end

    % Ensure kernel is a vector
    kernel = squeeze(kernel);
    if ~isvector(kernel)
        error('The kernel must be a vector.');
    end
    
    % Determine the lengths of input cell array and kernel
    k1 = length(iptCell);
    k2 = length(kernel);
    len = min(k1, k2);
    
    % Initialize output matrix
    if isempty(iptCell{headPointer})
        optMatrix = [];
        return
    else
        optMatrix = zeros(size(iptCell{headPointer}));
    end
    
    % Perform temporal convolution
    for t = 1:len
        j = mod(headPointer - t, k1) + 1;
        if abs(kernel(t)) > 1e-16 && ~isempty(iptCell{j})
            optMatrix = optMatrix + iptCell{j} * kernel(t);
        end
    end
end
