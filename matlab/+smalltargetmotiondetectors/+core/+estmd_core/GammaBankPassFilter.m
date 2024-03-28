classdef GammaBankPassFilter < smalltargetmotiondetectors.core.BaseCore
    % GammaBankPassFilter Gamma bank pass filter
    %   This class implements a gamma bank pass filter with two gamma
    %   delays.
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        hGammaDelay1;  % Gamma delay 1 object
        hGammaDelay2;  % Gamma delay 2 object
    end


    properties(Hidden)
        hCellInput;
    end


    methods
        function self = GammaBankPassFilter()
            % Constructor method
            % Initializes the GammaBankPassFilter object

            self = self@smalltargetmotiondetectors.core.BaseCore();

            import smalltargetmotiondetectors.core.*;

            % Initialize gamma delays with specified parameters
            self.hGammaDelay1 = GammaDelay(2, 3);
            self.hGammaDelay2 = GammaDelay(3, 6);
            self.hCellInput = CellRecording();
        end
    end

    methods
        function init(self)
            % Initialization method
            % Initializes gamma delays
            if self.hGammaDelay1.tau < self.hGammaDelay2.tau
                self.hGammaDelay2.tau = self.hGammaDelay1.tau + 1;
            end

            self.hGammaDelay1.init(false);  % Initialize gamma delay 1
            self.hGammaDelay2.init(false);  % Initialize gamma delay 2

            if isempty(self.hCellInput.lenCell)
                self.hCellInput.lenCell = max(...
                    self.hGammaDelay1.lenKernel,...
                    self.hGammaDelay2.lenKernel);
            end
            self.hCellInput.init()
        end

        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Processes the input matrix using gamma delays
            
            self.hCellInput.push(iptMatrix);

            % Compute outputs of gamma delays
            gamma1Output = self.hGammaDelay1.process_cell(...
                self.hCellInput.cellData);
            gamma2Output = self.hGammaDelay2.process_cell(...
                self.hCellInput.cellData);
            % Compute the difference between the outputs
            optMatrix = gamma1Output - gamma2Output;
        end
    end

end
