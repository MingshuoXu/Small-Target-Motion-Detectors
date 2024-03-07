classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula class for motion detection
    %   This class represents the lobula layer in the DSTMD
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
       alpha1 = 3;  % Alpha parameter
       thetaList = [0:7] * pi / 4;  % List of theta values
    end

    properties(Hidden)
        hLateralInhi;  % Lateral inhibition component
        hDirectionInhi;  % Directional inhibition component
    end

    methods
        function self = Lobula()
            % Constructor method
            % Initializes the Lobula object
            
            self = self@smalltargetmotiondetectors.core.BaseCore();
            import smalltargetmotiondetectors.core.*;
            import smalltargetmotiondetectors.core.dstmd_core.*;
            
            self.hLateralInhi = SurroundInhibition();
            self.hDirectionInhi = DirectionInhi();   
        end

        function init(self)
            % Initialization method
            % Initializes the lateral and directional inhibition components
            
            self.hLateralInhi.init();
            self.hDirectionInhi.init();
        end

        function lobulaOpt = process(self, lobulaIpt)
            % Processing method
            % Performs motion processing on the input
            
            tm3Signal = lobulaIpt{1};
            mi1Delay4Signal = lobulaIpt{2};
            tm1Delay5Signal = lobulaIpt{3};
            tm1Delay6Signal = lobulaIpt{4};

            [imgH, imgW] = size(tm3Signal);
            numDict = length(self.thetaList);

            % Correlation range
            corrRow = (1+self.alpha1):(imgH-self.alpha1);
            CorrCol = (1+self.alpha1):(imgW-self.alpha1);
            
            % Correlation Output
            corrOutput = cell(numDict, 1);
            for idx = 1:numDict
                corrOutput{idx} = zeros(imgH, imgW);
            end
            countTheta = 1;
            for theta = self.thetaList
                % Correlation position
                X_Com = round(self.alpha1 * cos(theta + pi/2 ));
                Y_Com = round(self.alpha1 * sin(theta + pi/2 ));
                
                % Calculate correlation output
                corrOutput{countTheta}(corrRow, CorrCol)...
                    = tm3Signal(corrRow, CorrCol)...
                    .* ( ...
                    tm1Delay5Signal(corrRow, CorrCol)...
                    + mi1Delay4Signal(corrRow-X_Com, CorrCol-Y_Com)...
                    ) ...
                    .* tm1Delay6Signal(corrRow-X_Com, CorrCol-Y_Com);
                countTheta = countTheta + 1;
            end

            % Perform lateral inhibition
            lateralInhiOpt = cell(numDict, 1);
            for idx = 1:numDict
                lateralInhiOpt{idx} = ...
                    self.hLateralInhi.process(corrOutput{idx});
            end

            % Perform directional inhibition
            lobulaOpt = self.hDirectionInhi.process(lateralInhiOpt); 
            self.Opt = lobulaOpt;
        end
    end
end
