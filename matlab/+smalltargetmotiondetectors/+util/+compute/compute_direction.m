function directionOpt = compute_direction(ipt)
    % Compute the dominant direction given a set of directional responses
    
    % Get the number of directions
    numDirection = length(ipt);
    
    % Get the size of the input matrix
    [m, n] = size(ipt{1});
    
    % Initialize variables for cosine and sine components of the direction
    % response
    outputCos = zeros(m, n);
    outputSin = zeros(m, n);
    
    % Compute the weighted sum of cosine and sine components for each
    % direction
    for idx = 1:numDirection
        outputCos = ...
            outputCos + ipt{idx} * cos((idx - 1) * 2 * pi / numDirection);
        outputSin = ...
            outputSin + ipt{idx} * sin((idx - 1) * 2 * pi / numDirection);
    end
    
    % Compute the direction based on the atan2 function
    directionOpt = atan2(outputSin, outputCos);
    
    % Adjust directions to be in the range [0, 2*pi]
    directionOpt(directionOpt < 0) = directionOpt(directionOpt < 0) + 2 * pi;
    
    % Set directions where both sine and cosine components are zero to NaN
    nonId = ~(logical(outputSin) & logical(outputCos));
    directionOpt(nonId) = nan;
end
