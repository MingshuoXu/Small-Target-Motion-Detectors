classdef Stmdcell < ...
        smalltargetmotiondetectors.core.BaseCore
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
            import smalltargetmotiondetectors.core.*;

            self.hGammaDelay = GammaDelay(6, 12);
            % Initialize the SurroundInhibition component
            self.hSubInhi = SurroundInhibition();

            self.cellDPlusE = CellRecording();
        end
    end

    methods
        function init(self)
            % Initialization method
            % This method initializes the Lobula layer component
            
            import smalltargetmotiondetectors.tool.kernel.*;
            
            % Initialize parameters and kernels
            self.hSubInhi.init();

            self.hGammaDelay.init();

            if isempty(self.cellDPlusE.lenCell)
                self.cellDPlusE.lenCell = self.hGammaDelay.lenKernel;
            end
            self.cellDPlusE.init();

        end

        function lateralInhiSTMDOpt = process(self, ...
                tm3Signal, tm1Signal, faiList, psiList)
            % Processing method
            % Performs temporal convolution, correlation, and surround
            % inhibition

            import smalltargetmotiondetectors.tool.compute.*;
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;
            %% Formula (11)
            convnIpt = cell(self.cellDPlusE.lenCell, 1);

            for idxT = 1:length(convnIpt)
                if ~isempty(self.cellDPlusE.cellData{idxT})

                    fai = faiList(idxT);
                    psi = psiList(idxT);
                    if ~isempty(self.cellDPlusE.cellData{idxT})
                        convnIpt{idxT} = slice_matrix_holding_size(...
                            self.cellDPlusE.cellData{idxT},...
                            psi,...
                            fai);
                    end

                end
            end

            feedbackSignal = self.alpha * ...
                self.hGammaDelay.process_cell(convnIpt);


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
            self.cellDPlusE.push(correlationD + correlationE);

            % Store the output in Opt property
            self.Opt = lateralInhiSTMDOpt;
        end


    end

end
