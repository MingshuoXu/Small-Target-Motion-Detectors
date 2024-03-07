classdef MatrixNMS < handle
    %MATRIXNMS Non-Maximum Suppression in Matrix
    %
    %   Properties:
    %       - maxRegionSize: The size of the region for maximum operation.
    %       - method: The method used for non-maximum suppression.
    %
    %   Methods:
    %       - nms: Performs non-maximum suppression on the input matrix.
    %       - select_auto_method: Automatically selects the method based on input matrix size.
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
        hasAutoMethod = false;
        autoMethod;
    end

    methods
        function self = MatrixNMS(maxRegionSize, method)
            % Constructor method
            if nargin < 1
                method = 'bubble';
                maxRegionSize = 5;
            elseif nargin == 1
                if maxRegionSize > 3
                    method = 'bubble';
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
        outputMartix = nms(self, inputMatrix);
        autoMethod = select_auto_mothod(self, inputMatrix);
        outputMartix = sort_nms(self, inputMatrix);
        outputMartix = conv2_nms(self, inputMatrix);
        outputMartix = bubble_nms(self, inputMatrix);
        outputMartix = greedy_nms(self, inputMatrix);
    end

    methods(Static)
        % Static methods
        function varargout = mappingAutoMethod(mapKey, addmapValue)
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
            else
                mapAutoMethod(mapKey) = addmapValue;
                varargout = {};
            end
        end
    end

end
