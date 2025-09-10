classdef MatrixNMS < handle
    %MATRIXNMS Non-Maximum Suppression in Matrix
    %
    %   Properties:
    %       - maxRegionSize: The size of the region for maximum operation.
    %       - method: The method used for non-maximum suppression.
    %
    %   Methods:
    %       - nms: Performs non-maximum suppression on the input matrix.
    %       - select_auto_method: Automatically selects the method based on
    % input matrix size.
    %       - sort_nms: Performs non-maximum suppression using sorting method.
    %       - conv2_nms: Performs non-maximum suppression using conv2 method.
    %       - bubble_nms: Performs non-maximum suppression using bubble method.
    %       - greedy_nms: Performs non-maximum suppression using greedy method.
    %       - mappingAutoMethod: Maps auto method based on key-value pairs.

    properties
        maxRegionSize;
        method;
    end

    properties(Hidden)
        nullAutoMethod = true;
        autoMethod;
    end

    methods
        function self = MatrixNMS(maxRegionSize, method)
            % Constructor method
            if nargin < 1
                method = 'sort';
                maxRegionSize = 5;
            elseif nargin == 1
                if maxRegionSize > 3
                    method = 'sort';
                else
                    method = 'conv2';
                end
            end
            self.maxRegionSize = maxRegionSize;
            self.method = method;
        end
    end

    methods
        % Public methods
        function outputMartix = nms(self, inputMatrix)
            % Performs non-maximum suppression based on the selected method
            maxRS = self.maxRegionSize;

            if strcmp(self.method, 'auto')
                if self.nullAutoMethod
                        % If auto method is not determined yet
                        [M, N] = size(inputMatrix);
                        % Determine auto method based on input matrix size
                        self.autoMethod = self.mapping_auto_method(...
                            sprintf('%d-%d-%d', M, N, maxRS)...
                            );
                        if isempty(self.autoMethod)
                            % Select auto method if not determined before
                            self.select_auto_method(inputMatrix);
                            % Save auto method in mapping
                            self.mapping_auto_method( ...
                                sprintf('%d-%d-%d', M, N, maxRS),...
                                self.autoMethod);
                        end
                        self.nullAutoMethod = false;

                    end

            else
                self.autoMethod = self.method;
            end

            switch self.autoMethod
                case 'conv2'
                    outputMartix = self.conv2_nms(inputMatrix, maxRS);
                
                case 'sort'
                    outputMartix = self.sort_nms(inputMatrix, maxRS);

                case 'greedy'
                    outputMartix = self.greedy_nms(inputMatrix, maxRS);

                case 'bubble'
                    outputMartix = self.bubble_nms(inputMatrix, maxRS);                   

                otherwise
                    % Error for invalid method
                    error('method must be ''sort'', ''bubble'', ''conv2'', or ''greedy''');

            end % End of Switch
        end

        function select_auto_mothod(self, inputMatrix)
            % Selects the most efficient method for non-maximum suppression automatically

            nTimes = 3;
            maxRS = self.maxRegionSize;
            % Measure the time taken by each method and repeat the process multiple times
            for methodIdx = 1:4
                switch methodIdx
                    case 1
                        tic;
                        for iT = 1:nTimes
                            if iT == 2
                                timeTic = tic;
                            end
                            [~] = self.bubble_nms(inputMatrix, maxRS);
                        end
                        timeBubble = toc(timeTic);
                    case 2
                        tic;
                        for iT = 1:nTimes
                            if iT == 2
                                timeTic = tic;
                            end
                            [~] = self.conv2_nms(inputMatrix, maxRS);
                        end
                        timeConv2 = toc(timeTic);
                    case 3
                        tic;
                        for iT = 1:nTimes
                            if iT == 2
                                timeTic = tic;
                            end
                            [~] = self.greedy_nms(inputMatrix, maxRS);
                        end
                        timeGreedy = toc(timeTic);
                    case 4
                        tic;
                        for iT = 1:nTimes
                            if iT == 2
                                timeTic = tic;
                            end
                            [~] = self.sort_nms(inputMatrix, maxRS);
                        end
                        timeSort = toc(timeTic);
                end % end switch
            end % end for

            % Determine the fastest method
            [~, fastIndex] = min([timeBubble, timeConv2, timeGreedy, timeSort]);

            % Map the index to the corresponding method name
            switch fastIndex
                case 1
                    self.autoMethod = 'bubble';
                case 2
                    self.autoMethod = 'conv2';
                case 3
                    self.autoMethod = 'greedy';
                case 4
                    self.autoMethod = 'sort';
                otherwise
                    error('Error: Invalid method index.');
            end

        end % [EoF]


    end % end of methods 

    methods(Static) % Static methods

        function outputMartix = conv2_nms(inputMatrix, maxRegionSize)
            % Performs non-maximum suppression using conv2 method

            [M, N] = size(inputMatrix);
            outputMartix = inputMatrix; % Initialize output matrix

            % Iterate over the region defined by maxRegionSize
            for rr = -maxRegionSize:maxRegionSize
                for cc = -maxRegionSize:maxRegionSize
                    % Define the regions to extract from the input matrix
                    rr1 = max(1, 1 + rr) : min(M, M + rr);
                    cc1 = max(1, 1 + cc) : min(N, N + cc);
                    rr2 = max(1, 1 - rr) : min(M, M - rr);
                    cc2 = max(1, 1 - cc) : min(N, N - cc);

                    % Extract the submatrices from the input/output matrix
                    temp = outputMartix(rr2, cc2);
                    subMatrix = inputMatrix(rr1, cc1);

                    % Perform element-wise comparison to update the output 
                    outputMartix(rr2, cc2) = temp .* (temp >= subMatrix);
                end % end for cc
            end % end for rr
        end

        function outputMartix = sort_nms(inputMatrix, maxRegionSize)
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
                        max(1, x-maxRegionSize) : min(M, x+maxRegionSize);
                    internalY = ...
                        max(1, y-maxRegionSize) : min(N, y+maxRegionSize);
                    isNotSupp(internalX, internalY) = false;

                    % Check if the current pixel is the maximum 
                    %   in its local region
                    maxLocalValue = max(...
                        inputMatrix(internalX,internalY), [], 'all'...
                        );
                    if inputMatrix(x,y) == maxLocalValue
                        isNotSupp(x,y) = true;
                    end
                end
            end

            % Apply non-maximum suppression
            outputMartix = inputMatrix .* isNotSupp;

        end % [EoF]

        function outputMartix = greedy_nms(inputMatrix, maxRegionSize)
            % Performs non-maximum suppression using the greedy method

            [M, N] = size(inputMatrix);
            iptMatrix = inputMatrix;
            isNotSupp = true(M, N);

            globalM = M;
            globalN = N;
            maxRS = maxRegionSize;
            opMartix = zeros(M, N);

            % Iterate over each element in the input matrix
            for x = 1:M
                for y = 1:N
                    if isNotSupp(x, y)
                        findMax2Supp(x, y);
                    end
                end % end for y
            end % end for x

            outputMartix = opMartix;

            % sub-function ---------- %
            function findMax2Supp(x, y)
                % Define a nested function to recursively find
                %   the maximum and suppress
                gInternalX = max(1, x - maxRS) : min(globalM, x + maxRS);
                gInternalY = max(1, y - maxRS) : min(globalN, y + maxRS);

                [~, globalMaxIndex] = max(...
                    iptMatrix(gInternalX, gInternalY), [], 'all'...
                    );

                [maxIdX0, maxIdY0] = ind2sub(...
                    [min(globalM, x + maxRS) - max(1, x - maxRS) + 1, ...
                    min(globalN, y + maxRS) - max(1, y - maxRS) + 1], ...
                    globalMaxIndex);

                maxIdX = maxIdX0 + max(1, x - maxRS) - 1;
                maxIdY = maxIdY0 + max(1, y - maxRS) - 1;

                internalMaxIdX = ...
                    max(1, maxIdX - maxRS) : min(globalM, maxIdX + maxRS);
                internalMaxIdY = ...
                    max(1, maxIdY - maxRS) : min(globalN, maxIdY + maxRS);

                if maxIdX == x && maxIdY == y
                    opMartix(maxIdX, maxIdY) = iptMatrix(maxIdX, maxIdY);
                    isNotSupp(internalMaxIdX, internalMaxIdY) = false;
                else
                    if isNotSupp(maxIdX, maxIdY)
                        findMax2Supp(maxIdX, maxIdY);
                    end
                    if maxIdX < x || (maxIdX == x && maxIdY < y)
                        isNotSupp(...
                            intersect(gInternalX, internalMaxIdX), ...
                            intersect(gInternalY, internalMaxIdY)...
                            ) = false;
                    end
                end
            end % end of sub funciton

        end % [EoF]

        function outputMartix = bubble_nms(inputMatrix, maxRS)
            % Performs non-maximum suppression using bubble method

            [M, N] = size(inputMatrix);
            outputMartix = zeros(M, N); % Initialize output matrix
            copyIptMatrix = inputMatrix; % Create a copy of input matrix

            % Find the maximum value and its index in the input matrix
            [maxValue, maxIndex] = max(copyIptMatrix, [], 'all');

            % Continue the process until maxValue drops below a threshold
            while maxValue > 1e-16
                % Convert linear index of the maximum value to subscripts
                [x, y] = ind2sub([M, N], maxIndex);

                % Define the region around the maximum value
                internalX = max(1, x - maxRS) : min(M, x + maxRS);
                internalY = max(1, y - maxRS) : min(N, y + maxRS);

                % Find the maximum value within the defined region
                maxLocalValue = max(...
                    inputMatrix(internalX, internalY), [], 'all'...
                    );

                % Check if the maximum value in the region is equal to
                %   the maximum value in the input matrix
                if maxValue == maxLocalValue
                    % Set the output matrix value to the maximum value
                    outputMartix(x, y) = maxLocalValue;
                end

                % Set the values in the defined region of the
                %   copy input matrix to zero
                copyIptMatrix(internalX, internalY) = 0;

                % Find the new maximum value and its index in the copy 
                %   input matrix
                [maxValue, maxIndex] = max(copyIptMatrix, [], 'all');
            end
        end % [EoF]

        function varargout = mapping_auto_method(mapKey, addmapValue)
            % Maps auto method based on key-value pairs
            persistent mapAutoMethod;
            if isempty(mapAutoMethod)
                mapAutoMethod = containers.Map();
            end
            if nargin == 1
                if isKey(mapAutoMethod, mapKey)
                    mapValue = mapAutoMethod(mapKey);
                else
                    mapValue = [];
                end
                varargout = {mapValue};
            elseif nargin == 2
                mapAutoMethod(mapKey) = addmapValue;
                varargout = {};
            end
        end % [EoF]

    end % End of Method

end % End of Class
