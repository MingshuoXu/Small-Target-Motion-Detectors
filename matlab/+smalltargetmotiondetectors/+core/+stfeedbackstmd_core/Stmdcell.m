classdef Stmdcell < smalltargetmotiondetectors.core.BaseCore
    % Lobula layer of the motion detection system
    %   This class represents the Lobula layer of the motion detection
    %   system. It performs several operations including temporal
    %   convolution, correlation, and surround inhibition.
    %
    %   Date: 2024-03-18

    properties
        hSubInhi;  % SurroundInhibition component
        alpha = 0.1;  % Parameter alpha 
 
    end

    properties(Hidden)
        gaussKernel;  % Gaussian kernel
        hGammaDelay;
        cellDPlusE;
    end


    methods
        function self = Stmdcell()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.SurroundInhibition;
            import smalltargetmotiondetectors.core.math_operator.GammaDelay;
            import smalltargetmotiondetectors.util.CircularCell;

            self.hGammaDelay = GammaDelay(6, 12);
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();

            self.cellDPlusE = CircularCell();
        end
    end

    methods
        function init_config(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            import smalltargetmotiondetectors.util.kernel.*;
            
            % Initialize parameters and kernels
            self.hSubInhi.init_config();

            self.hGammaDelay.init_config();

            if isempty(self.cellDPlusE.len)
                self.cellDPlusE.len = self.hGammaDelay.lenKernel;
            end
            self.cellDPlusE.init_config();

        end

        function lateralInhiSTMDOpt = process(self, ...
                tm3Signal, tm1Signal, faiList, psiList)
            % Processing method
            % Performs temporal convolution, correlation, and surround
            % inhibition

            import smalltargetmotiondetectors.util.compute.*;
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;
            %% Formula (11)
            convnIpt = cell(self.cellDPlusE.len, 1);


            point = self.cellDPlusE.point;
            for idxT = length(convnIpt):-1:1
                if ~isempty(self.cellDPlusE.data{point})
                    fai = faiList(idxT);
                    psi = psiList(idxT);
                    if ~isempty(self.cellDPlusE.data{point})
                        convnIpt{idxT} = slice_matrix_holding_size(...
                            self.cellDPlusE.data{point},...
                            psi,...
                            fai);
                    end
                end

                if point == 1
                    point = self.cellDPlusE.len;
                else
                    point = point - 1;
                end
            end


            feedbackSignal = self.alpha * ...
                self.hGammaDelay.process(convnIpt);


            %% Formula (10)
            % There must have the max operation here
            if ~isempty(feedbackSignal)
                correlationD = ...
                    max( (tm3Signal - feedbackSignal), 0) .* ...
                    max( (tm1Signal - feedbackSignal), 0);
            else
                correlationD = ...
                    max( (tm3Signal), 0) .* max( (tm1Signal), 0);
            end
            
            % Formula (13)
            correlationE = ...
                conv2(tm3Signal.*tm1Signal, self.gaussKernel, 'same');

            % Formula (14)
            % lateralInhiSTMDopt: STMD output after lateral inhibition in formula (14)
            lateralInhiSTMDOpt = self.hSubInhi.process(correlationD);

            % only record correlationD + correlationE
            self.cellDPlusE.circrecord(correlationD + correlationE);

            % Store the output in Opt property
            self.Opt = lateralInhiSTMDOpt;
        end


    end

end
