classdef GammaDelay < smalltargetmotiondetectors.core.BaseCore
    % GAMMADELAY Gamma filter in lamina layer
    %   References:
    %   * S. D. Wiederman, P. A. Shoemarker, D. C. O'Carroll, A model
    %     for the detection of moving targets in visual clutter inspired by
    %     insect physiology, PLoS ONE 3 (7) (2008) e2784.
    %           Page 5: In addition to this step, a second subtractive 
    %           inhibition is applied based on the average of the surrounding 
    %           input signals of the same channel polarity (surrounding ON 
    %           subtractively inhibit the centre ON channel and similarly for 
    %           the OFF channels). This is based on the surround inhibitory 
    %           effect found in the 'on-off' cells [21].
    %
    %   * Wang H, Peng J, Yue S. A directionally selective small target
    %     motion detecting visual neural network in cluttered backgrounds[J].
    %     IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    %
    %   Author: Mingshuo Xu
    %   Date: 2022-01-10
    %   LastEditTime: 2024-02-22

    properties
        order;      % Order of the gamma filter
        tau;        % Time constant of the filter
        lenKernel;  % Length of the filter kernel
        
        isRecord = true;
        isInLoop = false; % no circle the point in CircularCell
    end

    
    properties (Hidden)
        gammaKernel;    % Gamma filter kernel
        hCellInput;  % Input history stored in a cell array
    end
    

    methods
        function self = GammaDelay(order, tau, lenKernal)
            % Constructor
            % Initializes the GammaDelay object with provided parameters
            self = self@smalltargetmotiondetectors.core.BaseCore();
            
            if nargin >= 1
                self.order = order;
            end
            if nargin >= 2
                self.tau = tau;
            end
            if nargin >= 3
                self.lenKernel = lenKernal;
            end
        end

        function init_config(self, isRecord)
            % Initialization method
            % Creates the gamma filter kernel and initializes input history
            import smalltargetmotiondetectors.util.kernel.*;
            import smalltargetmotiondetectors.core.*;
            import smalltargetmotiondetectors.util.*;

            if nargin > 1
                self.isRecord = isRecord;
            end

            if self.order < 1
                self.order = 1;
            elseif ~isinteger(self.order)
                self.order = round(self.order);
            end

            if isempty(self.lenKernel)
                self.lenKernel = ceil(3 * self.tau);
            end

            if isempty(self.gammaKernel)
                self.gammaKernel = create_gamma_kernel( ...
                    self.order, ...
                    self.tau, ...
                    self.lenKernel);
            end

            if self.isRecord
                self.hCellInput = CircularCell(self.lenKernel);
                self.hCellInput.init_config();
            end
        end

        function optMatrix = process(self, input)
            % Processing method
            % Applies the gamma filter to the input matrix

            if isa(input, 'smalltargetmotiondetectors.util.CircularCell')
                optMatrix = self.process_circularcell(input);
            elseif iscell(input)
                optMatrix = self.process_cell(input);
            elseif ismatrix(input)
                optMatrix = self.process_matrix(input);
            end
        end

        function optMatrix = process_matrix(self, iptMatrix)
            if self.isInLoop
                self.hCellInput.cover(iptMatrix);
            else
                self.hCellInput.circrecord(iptMatrix);
            end
            
            optMatrix = self.process_circularcell(self.hCellInput);
        end

        function optMatrix = process_cell(self, iptCell)
            import smalltargetmotiondetectors.util.compute.compute_temporal_conv;

            optMatrix = compute_temporal_conv(...
                iptCell, ...
                self.gammaKernel, ...
                length(iptCell));
        end

        function optMatrix = process_circularcell(self, circularCellObj)
            import smalltargetmotiondetectors.util.compute.compute_circularcell_conv;
            optMatrix = compute_circularcell_conv(...
                circularCellObj, ...
                self.gammaKernel);
        end

    end
end
