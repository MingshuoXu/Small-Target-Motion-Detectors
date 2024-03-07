function outputMartix = sort_nms(self, inputMatrix)
    % Performs non-maximum suppression using sorting method

    [M, N] = size(inputMatrix);
    [indexI, indexJ, valueInputMatrix] = find(inputMatrix);
    [~, indexSub] = sort(valueInputMatrix, 'descend');
    indexI = indexI(indexSub);
    indexJ = indexJ(indexSub);

    n = length(indexI);
    isNotSupp = true(M,N);

    % Iterate through sorted indices
    for ii = 1:n
        x = indexI(ii);
        y = indexJ(ii);
        if isNotSupp(x,y)
            internalX = ...
                max(1, x-self.maxRegionSize) : min(M, x+self.maxRegionSize);
            internalY = ...
                max(1, y-self.maxRegionSize) : min(N, y+self.maxRegionSize);
            isNotSupp(internalX, internalY) = false;

            % Check if the current pixel is the maximum in its local region
            maxLocalValue = max(inputMatrix(internalX,internalY), [], 'all');
            if inputMatrix(x,y) == maxLocalValue
                isNotSupp(x,y) = true;
            end
        end
    end

    % Apply non-maximum suppression
    outputMartix = inputMatrix .* isNotSupp;

end
