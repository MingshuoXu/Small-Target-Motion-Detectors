classdef MushroomBody < smalltargetmotiondetectors.core.BaseCore
    % MushroomBody class for STMDPlus

    properties
        maxLenRecordContrast = 100; % Length of Record
        SDThres = 5; % Threshold of standard deviation
    end

    properties(Hidden)
        contrastRecord = {}; % trackInfo data
    end


    methods
        function self = MushroomBody()
            % Constructor method
            % Initializes the MushroomBody object

            self = self@smalltargetmotiondetectors.core.BaseCore();

        end
    end

    methods
        function init_config(self)
            % Initialization method
            % Initializes the non-maximum suppression
        end

        function mushroomBodyOpt = process(self, ...
                sTrajectory, lobulaOpt, contrastOpt)
            % Processing method
            % Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt

            mushroomBodyOpt = lobulaOpt;

            if isempty(sTrajectory)
                self.contrastRecord = {};
                return;
            end

            cellTrajectory = struct2cell(sTrajectory);
            cellLocation = cellTrajectory(1,1,:);

            lenT = length(sTrajectory);

            newContrastRecord = cell(lenT, 1);
            
            numContrast = length(contrastOpt);
            

            for newId = 1:lenT
                oldId = sTrajectory(newId).oldId;

                % lost tract
                if sTrajectory(newId).lostCount > 1
                    newContrastRecord{newId} = self.contrastRecord{oldId};
                    continue;
                end

                % contrast of new index
                xNew = cellLocation{newId}(1);
                yNew = cellLocation{newId}(2);
                nowContrast = zeros(numContrast, 1);
                for idCont = 1:numContrast
                    nowContrast(idCont, 1) = ...
                        contrastOpt{idCont}(xNew, yNew);
                end


                if isnan(oldId)
                    % new response
                    newContrastRecord{newId} = nowContrast;
                else
                    if length(self.contrastRecord{oldId}) >= ...
                            self.maxLenRecordContrast
                        newContrast_newId = ...
                            circshift(self.contrastRecord{oldId}, [0, -1]);
                        newContrast_newId(:, end) = nowContrast;
                        newContrastRecord{newId} = newContrast_newId;
                    else
                        newContrastRecord{newId} = ...
                            [self.contrastRecord{oldId}, nowContrast];
                    end

                    %% Small Target Discrimination
                    if max(std(newContrastRecord{newId}, 0, 2)) < self.SDThres
                        mushroomBodyOpt(xNew, yNew) = 0;
                    end

                end

            end

            self.contrastRecord = newContrastRecord;

            self.Opt = mushroomBodyOpt;

        end
    end

end
