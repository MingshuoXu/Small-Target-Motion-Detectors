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
        function init(self)
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

            newContrast = cell(lenT, 1);

            for newId = 1:lenT
                oldId = sTrajectory(newId).oldId;

                % lost tract
                if sTrajectory(newId).lostCount > 1
                    newContrast{newId} = self.contrastRecord{oldId};
                    continue;
                end

                % contrast of new index
                xNew = cellLocation{newId}(1);
                yNew = cellLocation{newId}(2);
                nowContrast = squeeze(contrastOpt(xNew, yNew, :));


                if isnan(oldId)
                    % new response
                    newContrast{newId} = nowContrast;
                else
                    if length(self.contrastRecord{oldId}) >= ...
                            self.maxLenRecordContrast
                        newContrast{newId} = [...
                            self.contrastRecord{oldId}(:,2:end),...
                            nowContrast ];
                    else
                        newContrast{newId} = [...
                            self.contrastRecord{oldId},...
                            nowContrast ];
                    end

                    %% Small Target Discrimination
                    if max(std(newContrast{newId}, 0, 2)) < self.SDThres
                        mushroomBodyOpt(xNew, yNew) = 0;
                    end

                end

            end

            self.contrastRecord = newContrast;

            self.Opt = mushroomBodyOpt;

        end
    end

end
