classdef Lobula < smalltargetmotiondetectors.core.BaseCore
    % Lobula class for motion detection
    %   This class represents the lobula layer in the DSTMD
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
       alpha1 = 3;  % Alpha parameter
       thetaList = (0:7) * pi / 4;  % List of theta values
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
            import smalltargetmotiondetectors.core.math_operator.*;
            import smalltargetmotiondetectors.core.dstmd_core.*;
            
            self.hLateralInhi = SurroundInhibition();
            self.hDirectionInhi = DirectionInhi();   
        end

        function init_config(self)
            % Initialization method
            % Initializes the lateral and directional inhibition components
            
            self.hLateralInhi.init_config();
            self.hDirectionInhi.init_config();
        end

        function lobulaOpt = process(self, lobulaIpt)
            % Processing method
            % Performs motion processing on the input

            tm3Signal = lobulaIpt{1};
            mi1Para4Signal = lobulaIpt{2};
            tm1Para5Signal = lobulaIpt{3};
            tm1Para6Signal = lobulaIpt{4};

            [imgH, imgW] = size(tm3Signal);
            numDict = length(self.thetaList);

            % Correlation range
            xRange = (1+self.alpha1):(imgH-self.alpha1);
            yRange = (1+self.alpha1):(imgW-self.alpha1);
            
            % Correlation Output
            correOutput = cell(numDict, 1);
            for idx = 1:numDict
                correOutput{idx} = zeros(imgH, imgW);
            end
            countTheta = 1;
            for theta = self.thetaList
                % Correlation position
                shiftX = round(self.alpha1 * cos(theta + pi/2 ));
                shiftY = round(self.alpha1 * sin(theta + pi/2 ));
                
                % Calculate correlation output
                correOutput{countTheta}(xRange, yRange)...
                    = tm3Signal(xRange, yRange)...
                    .* ( ...
                    tm1Para5Signal(xRange, yRange)...
                    + mi1Para4Signal(xRange-shiftX, yRange-shiftY)...
                    ) ...
                    .* tm1Para6Signal(xRange-shiftX, yRange-shiftY);
                countTheta = countTheta + 1;
            end

            % Perform lateral inhibition
            lateralInhiOpt = cell(numDict, 1);
            for idx = 1:numDict
                lateralInhiOpt{idx} = ...
                    self.hLateralInhi.process(correOutput{idx});
            end

            % Perform directional inhibition
            lobulaOpt = self.hDirectionInhi.process(lateralInhiOpt); 
            self.Opt = lobulaOpt;
        end
    end
end
