classdef MushroomBody < smalltargetmotiondetectors.core.BaseCore
    % MushroomBody class for STMDPlus

    properties
        paraNMS = struct( ...
            'maxRegionSize', 15, ...
            'method', 'sort'); % Parameters for non-maximum suppression

        detectThreshold = 0.01; % Response threshold for clustering
        DBSCANDist = 5; % Spatial distance for clustering
        lenDBSCAN = 100; % Length of clustering trajectory
        SDThreshold = 5; % Threshold of standard deviation
    end

    properties(Hidden)
        T1Kernel; % T1 kernel
        hNMS; % object's handle of non-maximum suppression
        trackID; % trackInfo index
        trackInfo = {}; % trackInfo data
        hasFunPdist2; % check if there is a build-in function named pdist2
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
            
            import smalltargetmotiondetectors.tool.MatrixNMS;

            self.hNMS = MatrixNMS( ...
                self.paraNMS.maxRegionSize, ...
                self.paraNMS.method);

            % Check if pdist2 is available
            if exist('pdist2') == 2 %#ok<EXIST>
                % Use the built-in pdist2 function
                self.hasFunPdist2 = true;
            else
                % Use the compute_pdist2 function
                self.hasFunPdist2 = false;
            end


        end
        
        function mushroomBodyOpt = process(self, lobulaOpt, contrastOpt)
            % Processing method
            % Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt
            
            import smalltargetmotiondetectors.tool.compute.*;

            maxLobulaOpt = compute_response(lobulaOpt);
            nmsLobulaOpt = self.hNMS.nms(maxLobulaOpt);

            numDirection = length(lobulaOpt);
            mushroomBodyOpt = cell(numDirection, 1);
            for idxI = 1 : numDirection
                mushroomBodyOpt{idxI} = ...
                    lobulaOpt{idxI} .* logical(nmsLobulaOpt);
            end
            maxNumber = max(nmsLobulaOpt, [], 'all');

            if maxNumber <= 0
                self.trackID = [];
                self.trackInfo = {};
                return;
            end

            [idX, idY] = find(nmsLobulaOpt > self.detectThreshold * maxNumber);
            newID = [idX, idY];

            shouldTrackID = true(size(self.trackID,1), 1);
            shouldAddNewID = true(length(idX), 1);
            
            %% Information Integration -- join
            if ~isempty(self.trackID)

                if self.hasFunPdist2
                    DD = pdist2(self.trackID, newID);
                else
                    DD = compute_pdist2(self.trackID, newID);
                end

                [D1, ind1] = min(DD, [], 2);

                for idxI = 1:length(D1)
                    if D1(idxI) <= self.DBSCANDist
                        idxJ = ind1(idxI);
                        if shouldAddNewID(idxJ)
                            self.trackID(idxI,:) = newID(idxJ,:);
                            self.trackInfo{idxI} = [
                                self.trackInfo{idxI}, ...
                                squeeze(contrastOpt(newID(idxJ,1),newID(idxJ,2),:))...
                                ];
                            shouldTrackID(idxI) = false;
                            shouldAddNewID(idxJ) = false;
                        end
                        
                    end
                end

                %% Information Integration - wipe off
                idxK = find(shouldTrackID);
                self.trackID(idxK,:) = [];
                self.trackInfo(idxK) = [];

            end

            oldTractNum = length(self.trackInfo);

            %% New Track
            isxNew = find(shouldAddNewID);
            for kk = isxNew'
                self.trackID(end+1,:) = newID(kk,:);
                self.trackInfo{end+1} = squeeze( ...
                    contrastOpt(newID(kk,1), newID(kk,2), :) ...
                    ) ;
            end

            %% Small Target Discrimination

            for idx = 1:oldTractNum

                if max(std(self.trackInfo{idx}, 0, 2)) < self.SDThreshold
                    for idxDirection = 1 : numDirection
                        idX = self.trackID(idx,1);
                        idY = self.trackID(idx,2);
                        mushroomBodyOpt{idxDirection}(idX, idY) = 0;
                    end
                end

                % Limit the length to ensure that the contents are
                %   within reasonable limits
                if size(self.trackInfo{idx},2) > self.lenDBSCAN
                    self.trackInfo{idx} = self.trackInfo{idx}(:,2:end);
                end
            end

            self.Opt = mushroomBodyOpt;
            
        end
    end

end
