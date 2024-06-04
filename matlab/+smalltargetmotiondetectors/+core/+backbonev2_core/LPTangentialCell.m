classdef LPTangentialCell < smalltargetmotiondetectors.core.BaseCore
    % direction in backbonev2

    properties
        kernelSize = 3;
        kernelCos;
        kernelSin;
        lptcMatrix;
    end


    methods
        function self = LPTangentialCell(kernelSize)
            if nargin > 1
                self.kernelSize = kernelSize;
            end 
        end

        function init_config(self)
            self.kernelCos = zeros(self.kernelSize, self.kernelSize);
            self.kernelSin = zeros(self.kernelSize, self.kernelSize);
            halfKernel = floor(self.kernelSize / 2);
            for x = -halfKernel:halfKernel
                for y = -halfKernel:halfKernel
                    r = sqrt(x^2 + y^2);
                    if r == 0
                        continue;
                    end
                    self.kernelCos(y+halfKernel+1, x+halfKernel+1) = x / r;
                    self.kernelSin(y+halfKernel+1, x+halfKernel+1) = -y / r;
                end
            end
        end

        function direction = process(self, laminaOpt, onSignal, offSignal)
            % get indexes
            onDire = (onSignal>0) & (laminaOpt>0);
            offDire = (offSignal>0) & (laminaOpt<0);

            direMatrix = zeros(size(laminaOpt));
            % On direction
            direMatrix(onDire) = offSignal(onDire) ./ onSignal(onDire);
            % Off direction
            direMatrix(offDire) = onSignal(offDire) ./ offSignal(offDire);

            directionCos = filter2(self.kernelCos, direMatrix, 'same');
            directionSin = filter2(self.kernelSin, direMatrix, 'same');
            direction = atan2(directionSin, directionCos);
            % Adjust directions to be in the range [0, 2*pi]
            direction(direction < 0) = direction(direction < 0) + 2 * pi;
            
            self.lptcMatrix = direMatrix;
            self.Opt = struct(...
                'direction', direction, ...
                'lptcMatric', direMatrix);
        end % EoF

    end % EoM

end % EoC
