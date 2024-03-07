function outputMartix = conv2_nms(self, inputMatrix)
    % Performs non-maximum suppression using conv2 method
    
    [M, N] = size(inputMatrix);
    outputMartix = inputMatrix; % Initialize output matrix

    % Iterate over the region defined by maxRegionSize
    for rr = -self.maxRegionSize:self.maxRegionSize
        for cc = -self.maxRegionSize:self.maxRegionSize
            % Define the regions to extract from the input matrix
            rr1 = max(1, 1 + rr) : min(M, M + rr);
            cc1 = max(1, 1 + cc) : min(N, N + cc);
            rr2 = max(1, 1 - rr) : min(M, M - rr);
            cc2 = max(1, 1 - cc) : min(N, N - cc);

            % Extract the submatrices from the output matrix and input matrix
            temp = outputMartix(rr2, cc2);
            inputSubmatrix = inputMatrix(rr1, cc1);

            % Perform element-wise comparison and update the output matrix
            outputMartix(rr2, cc2) = temp .* (temp >= inputSubmatrix);
        end % end for cc
    end % end for rr
end
