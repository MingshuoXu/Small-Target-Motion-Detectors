function outputMartix = bubble_nms(self, inputMatrix)
    % Performs non-maximum suppression using bubble method
    
    [M, N] = size(inputMatrix);
    outputMartix = zeros(M, N); % Initialize output matrix
    copyIptMatrix = inputMatrix; % Create a copy of input matrix
    
    % Find the maximum value and its index in the input matrix
    [maxValue, maxIndex] = max(copyIptMatrix, [], 'all');
    
    % Continue the process until maxValue drops below a threshold
    while maxValue > 1e-16
        % Convert the linear index of the maximum value to subscripts
        [x, y] = ind2sub([M, N], maxIndex);
        
        % Define the region around the maximum value
        internalX = max(1, x - self.maxRegionSize) : min(M, x + self.maxRegionSize);
        internalY = max(1, y - self.maxRegionSize) : min(N, y + self.maxRegionSize);
        
        % Find the maximum value within the defined region
        maxLocalValue = max(inputMatrix(internalX, internalY), [], 'all');
        
        % Check if the maximum value in the region is equal to the maximum value in the input matrix
        if maxValue == maxLocalValue
            outputMartix(x, y) = maxLocalValue; % Set the output matrix value to the maximum value
        end
        
        % Set the values in the defined region of the copy input matrix to zero
        copyIptMatrix(internalX, internalY) = 0;
        
        % Find the new maximum value and its index in the copy input matrix
        [maxValue, maxIndex] = max(copyIptMatrix, [], 'all');
    end
end
