classdef Lptcell < ...
        smalltargetmotiondetectors.core.BaseCore
    %Lptcell Lobula Plate Tangential Cell
    %

    properties
        bataList = 2:2:18;
        thetaList = (0:7) * pi/ 4;
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
        function init_config(self, lenVelocity)
            self.velocity = zeros(lenVelocity, 1);

            lenBataList = length(self.bataList);
            lenthetaList = length(self.thetaList);
            % generate gauss distribution
            try
                gaussianDistribution = normpdf(-199:200, 1, 100/2);
            catch
                gaussianDistribution = custom_normpdf(-199:200, 1, 100/2);
            end
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
                tm1Signal, tm2Signal, tm3Signal, mi1Signal, tau5)
            import smalltargetmotiondetectors.core.stfeedbackstmd_core.*;
            import smalltargetmotiondetectors.util.compute.*;

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

                    sumLplcOpt = sum(ltlcOpt, 'all');

%                     [m,n] = size(tm3Signal);
%                     sumLplcOpt = sumLplcOpt  / (255^2);

                    sumLplcOptR(idBata,idTheta) = sumLplcOpt;

                    idTheta = idTheta + 1;
                end
                idBata = idBata + 1;
            end

            %% preferTheta
            [firingRate, preferTheta] = max(sumLplcOptR, [], 2);
            maxTheta = max(preferTheta);
            % firingRate = sumLplcOptR(:, maxTheta);

            %% background velocity

            % argmax_{v}(prod(- exp( |r-f|^{2} )))
            %   --> argmax_{v}(exp( -sum( |r-f|^{2}) ))
            %   --> argmin_{v}( sum( |r-f|^{2} ) )
            %   --> argmin_{v}( norm(r-f) )
            self.velocity(1) = [];
            [~, self.velocity(end+1)] = ...
                min( sum( (firingRate-self.tuningCurvef).^2, 1) ) ;
%             self.velocity(end) = self.velocity(end);% / tau5 / 2;

%             fprintf('%f\n', self.velocity(end) );

            sumV = zeros(size(self.velocity));
            for idV = 1:length(self.velocity)
                sumV(idV) = sum(self.velocity(idV:end), "all");
            end

            %%
            fai = sumV * cos(maxTheta);
            psi = sumV * sin(maxTheta);


            %%
            self.Opt = {fai, psi};
        end


    end

end

function y = custom_normpdf(x, mu, sigma)
    y = exp(-0.5 * ((x - mu) / sigma).^2) / (sigma * sqrt(2 * pi));
end
