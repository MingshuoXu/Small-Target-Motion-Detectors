classdef GammaBandPassFilter < smalltargetmotiondetectors.core.BaseCore
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
        function self = GammaBandPassFilter()
            % Constructor method
            % Initializes the GammaBankPassFilter object

            self = self@smalltargetmotiondetectors.core.BaseCore();

            import smalltargetmotiondetectors.core.math_operator.GammaDelay;
            import smalltargetmotiondetectors.util.CircularCell;

            % Initialize gamma delays with specified parameters
            self.hGammaDelay1 = GammaDelay(2, 3);
            self.hGammaDelay2 = GammaDelay(3, 6);
            self.hCellInput = CircularCell();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % Initializes gamma delays
            if self.hGammaDelay1.tau < self.hGammaDelay2.tau
                self.hGammaDelay2.tau = self.hGammaDelay1.tau + 1;
            end

            % Initialize gamma delay 1
            self.hGammaDelay1.init_config(false); 
            % Initialize gamma delay 2
            self.hGammaDelay2.init_config(false);  

            if isempty(self.hCellInput.len)
                self.hCellInput.len = max(...
                    self.hGammaDelay1.lenKernel,...
                    self.hGammaDelay2.lenKernel);
            end
            self.hCellInput.init_config();
        end

        function optMatrix = process(self, iptMatrix)
            % Processing method
            % Processes the input matrix using gamma delays
            
            self.hCellInput.circrecord(iptMatrix);

            % Compute outputs of gamma delays
            gamma1Output = self.hGammaDelay1.process_circularcell(...
                self.hCellInput);
            gamma2Output = self.hGammaDelay2.process_circularcell(...
                self.hCellInput);
            % Compute the difference between the outputs
            optMatrix = gamma1Output - gamma2Output;
        end
    end

end
