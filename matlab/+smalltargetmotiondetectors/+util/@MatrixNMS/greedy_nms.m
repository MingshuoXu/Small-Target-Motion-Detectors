function outputMartix = greedy_nms(self, inputMatrix)
    % Performs non-maximum suppression using the greedy method

    [M, N] = size(inputMatrix);
    iptMatrix = inputMatrix;
    isNotSupp = true(M, N);

    globalM = M;
    globalN = N;
    maxRS = self.maxRegionSize;
    opMartix = zeros(M, N);

    % Define a nested function to recursively find the maximum and suppress
    function findMax2Supp(x, y)
        gInternalX = max(1, x - maxRS) : min(globalM, x + maxRS);
        gInternalY = max(1, y - maxRS) : min(globalN, y + maxRS);

        [~, globalMaxIndex] = max(iptMatrix(gInternalX, gInternalY), [], 'all');

        [maxIdX0, maxIdY0] = ind2sub(...
            [min(globalM, x + maxRS) - max(1, x - maxRS) + 1, ...
            min(globalN, y + maxRS) - max(1, y - maxRS) + 1], ...
            globalMaxIndex);

        maxIdX = maxIdX0 + max(1, x - maxRS) - 1;
        maxIdY = maxIdY0 + max(1, y - maxRS) - 1;

        internalMaxIdX = max(1, maxIdX - maxRS) : min(globalM, maxIdX + maxRS);
        internalMaxIdY = max(1, maxIdY - maxRS) : min(globalN, maxIdY + maxRS);

        if maxIdX == x && maxIdY == y
            opMartix(maxIdX, maxIdY) = iptMatrix(maxIdX, maxIdY);
            isNotSupp(internalMaxIdX, internalMaxIdY) = false;
        else
            if isNotSupp(maxIdX, maxIdY)
                findMax2Supp(maxIdX, maxIdY);
            end
            if maxIdX < x || (maxIdX == x && maxIdY < y)
                isNotSupp(intersect(gInternalX, internalMaxIdX), ...
                    intersect(gInternalY, internalMaxIdY)) = false;
            end
        end
    end

    % Iterate over each element in the input matrix
    for x = 1:M
        for y = 1:N
            if isNotSupp(x, y)
                findMax2Supp(x, y);
            end
        end % end for y
    end % end for x

    outputMartix = opMartix;
end
