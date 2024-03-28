classdef Lptcell < ...
        smalltargetmotiondetectors.core.BaseCore
    %Lptcell Lobula Plate Tangential Cell
    %
    %   Author: [Your Name]
    %   Date: [Date]

    properties
        bataList = 2:2:18;
        thetaList = (0:7) * pi/ 4;
        inputFps;
    end


    properties(Hidden)
        velocity;
        tuningCurvef;
        
    end


    methods
        function self = Lptcell()
            % Constructor method
            % Initializes the Lobula object

            self = self@smalltargetmotiondetectors.core.BaseCore();

        end
    end

    methods
        function init(self, lenVelocity)
            self.velocity = zeros(lenVelocity, 1);

            lenBataList = length(self.bataList);
            lenthetaList = length(self.thetaList);
            % generate gauss distribution
            gaussianDistribution = normpdf(-199:200, 1, 100/2);
            % normlization
            gaussianDistribution = ...
                gaussianDistribution / max(gaussianDistribution);

            self.tuningCurvef = zeros(lenthetaList,lenthetaList*100+200);
            self.tuningCurvef(1, 1:300) = gaussianDistribution(101:400);
            self.tuningCurvef(lenBataList, end-300+1:end) = ...
                gaussianDistribution(1:300);
            for id = 2 : lenBataList - 1
                idRange = id*100-200+1 : id*100+200;
                self.tuningCurvef(id, idRange) = gaussianDistribution;
            end
        end

        function [fai, psi] = process(self, ...
                tm1Signal, tm2Signal, tm3Signal, mi1Signal)
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;

            lenBataList = length(self.bataList);
            lenthetaList = length(self.thetaList);
            sumLplcOptR = zeros(lenBataList, lenthetaList);

            idBata = 1;
            for bata = self.bataList
                idTheta = 1;
                for theta = self.thetaList

                    shiftX = round(bata * cos(theta + pi/2 ));
                    shiftY = round(bata * sin(theta + pi/2 ));
                    shiftMi1Signal = slice_matrix_holding_size(...
                        mi1Signal, shiftY, shiftX);
                    shiftTm1Signal = slice_matrix_holding_size(...
                        tm1Signal, shiftY, shiftX);

                    ltlcOpt = tm3Signal .* shiftMi1Signal ...
                        + tm2Signal .* shiftTm1Signal;

                    sumLplcOpt = sum(ltlcOpt, "all");

                    % [m,n] = size(tm3Signal);
                    % sumLplcOpt = sumLplcOpt /m/n/(255^2);

                    sumLplcOptR(idBata,idTheta) = sumLplcOpt;

                    idTheta = idTheta + 1;
                end
                idBata = idBata + 1;
            end

            %% preferTheta
            [~, preferTheta] = max(sumLplcOptR, [], 2);
            maxTheta = max(preferTheta);
            firingRate = sumLplcOptR(:, maxTheta);

            %% background velocity

            % argmax_{v}(prod(- exp( |r-f|^{2} )))
            %   --> argmax_{v}(exp( -sum( |r-f|^{2}) ))
            %   --> argmin_{v}( sum( |r-f|^{2} ) )
            %   --> argmin_{v}( norm(r-f) )
            self.velocity(1) = [];
            [~, self.velocity(end+1)] = ...
                min( sum( (firingRate-self.tuningCurvef).^2, 1) ) ;
            self.velocity(end) = self.velocity(end) / self.inputFps;

            % fprintf('%f\n', self.velocity(end)  );

            sumV = zeros(size(self.velocity));
            for idV = 1:length(self.velocity)
                sumV(idV) = sum(self.velocity(idV:end), "all");
            end

            %% have to check %%%%%%%%%%%%%%
            
            fai = sumV * cos(maxTheta);
            psi = sumV * sin(maxTheta);
            %% have to check %%%%%%%%%%%%%%

            %%
            self.Opt = {fai, psi};
        end


    end

end
