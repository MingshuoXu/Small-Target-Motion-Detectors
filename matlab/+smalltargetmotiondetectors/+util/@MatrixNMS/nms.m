function outputMartix = nms(self, inputMatrix)
    % Performs non-maximum suppression based on the selected method
    
    switch self.method
        case {'bubble', 'conv2', 'greedy', 'sort'}
            % Call the corresponding method based on the selected method
            outputMartix = feval(str2func([self.method, '_nms']), self, inputMatrix);
        case 'auto'
            % If auto method is selected
            if self.hasAutoMethod
                % If auto method is already determined, call the corresponding method
                outputMartix = feval(str2func([self.autoMethod, '_nms']), self, inputMatrix);
            else
                % If auto method is not determined yet
                [M, N] = size(inputMatrix);
                % Determine auto method based on input matrix size
                self.autoMethod = self.mapping_auto_method(sprintf('%d-%d-%d', M, N, self.maxRegionSize));
                if isempty(self.autoMethod)
                    % Select auto method if not determined before
                    self.autoMethod = self.select_auto_mothod(inputMatrix);
                    % Save auto method in mapping
                    self.mapping_auto_method( ...
                        sprintf('%d-%d-%d', M, N, self.maxRegionSize), self.autoMethod);
                end
                self.hasAutoMethod = true;
                % Call nms function recursively with auto method
                outputMartix = self.nms(inputMatrix);
            end
        otherwise
            % Error for invalid method
            error('method must be ''bubble'', ''conv2'', ''greedy'', or ''auto''');
    end
end
