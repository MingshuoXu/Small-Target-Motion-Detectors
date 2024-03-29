classdef DirectionCell < smalltargetmotiondetectors.core.BaseCore
    % direction in backbonev2

    properties
        paraNMS = struct( ...
            'maxRegionSize', 10, ...
            'method', 'sort'); % Parameters for non-maximum suppression

        detectThreshold = 0.01; % Response threshold for clustering
        DBSCANDist = 5; % Spatial distance for clustering
        lenDBSCAN = 10; % Length of clustering trajectory

        sTrajectory;
    end

    properties(Constant, Hidden)
        initSturct = struct( ...
            'location', [NaN,NaN], ...
            'oldId', NaN, ...
            'history', nan(6, 2), ...
            'direction', NaN, ...
            'velocity', NaN, ...
            'accuV', NaN, ...
            'lostCount', 1 );
    end

    properties(Hidden)
        hNMS; % object's handle of non-maximum suppression


        % trackID; % trackInfo index
        % trackInfo = {}; % trackInfo data
        hasFunPdist2; % check if there is a build-in function named pdist2
        numResponse;
        lostThreshold = 8;
    end


    methods
        function self = DirectionCell()
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

        function [mBodyResponse, mBodyDirection] = process(self, lobulaOpt)
            % Processing method
            % Processes the input lobulaOpt and contrastOpt to 
            %   generate mushroomBodyOpt

            import smalltargetmotiondetectors.tool.compute.*;
            import smalltargetmotiondetectors.core.backbonev2_core.*;

            mBodyResponse = self.hNMS.nms(lobulaOpt);
            mBodyDirection = nan(size(mBodyResponse));

            self.record_trajectory_id();

            [idX, idY] = find(mBodyResponse > 0);
            self.numResponse = length(idX);

            if ~self.numResponse
                self.sTrajectory = struct([]);
                return;
            end


            newIndex = [idX, idY];

            shouldTrackID = true(length(self.sTrajectory), 1);
            shouldAddNewID = true(self.numResponse, 1);

            %% Information Integration -- join
            if ~isempty(self.sTrajectory)
                cellTrajectory = ...
                    struct2cell(self.sTrajectory);

                trajectoryLocation = cell2mat(cellTrajectory(1,1,:));
                if size(trajectoryLocation, 3) == 1
                    trajectoryLocation = squeeze(trajectoryLocation);
                else
                    trajectoryLocation = squeeze(trajectoryLocation)';
                end

                if self.hasFunPdist2
                    DD = pdist2(trajectoryLocation, newIndex);
                else
                    DD = compute_pdist2(trajectoryLocation, newIndex);
                end

                [D1, ind1] = min(DD, [], 2);

                for idxI = 1:length(D1)
                    if D1(idxI) <= self.DBSCANDist
                        idxJ = ind1(idxI);
                        if shouldAddNewID(idxJ)

                            % location------------------------------------
                            self.sTrajectory(idxI).location = ...
                                newIndex(idxJ,:);

                            % history location----------------------------
                            self.sTrajectory(idxI).history = ...
                                circshift(self.sTrajectory(idxI).history, -1);
                            self.sTrajectory(idxI).history(end, :) = ...
                                newIndex(idxJ,:);

                            % get not non history for caculating direction
                            %   velocity and acceleration velocity
                            notNanHist = self.sTrajectory(idxI).history;
                            notNanHist(isnan(notNanHist(:,1)), :) = [];
                            % notNanHist = notNanHist(~isnan(notNanHist));

                            % direction-------------------------------
                            if size(notNanHist, 1) > 1
                                self.sTrajectory(idxI).direction = ...
                                    get_direction_by_multipoints(notNanHist);
                            end

                            % vVector = ...
                            %     self.sTrajectory(idxI).location - ...
                            %     notNanHist(end-1,:);
                            % if any(vVector)
                            %     self.sTrajectory(idxI).direction = ...
                            %         atan2( vVector(1), vVector(2) );
                            % else
                            %     self.sTrajectory(idxI).direction = NaN;
                            % end

                            % velocity -----------------------------------
                            % lastV = self.sTrajectory(idxI).velocity;

                            % nowV = self.sTrajectory(idxI).history ...
                            %     / self.sTrajectory(idxI).lostCount;
                            % lastV = self.sTrajectory(idxI).localV;
                            % self.sTrajectory(idxI).velocity = ...
                            %     (nowV + lastV) / 2;

                            % self.sTrajectory(idxI).localV = ...
                            %     norm(directionVector) ...
                            %     / self.sTrajectory(idxI).lostCount;    

                            % acceleration velocity
                            % self.sTrajectory(idxI).acceleration = ...
                            %   self.sTrajectory(idxI).velocity - lastV;

                            % lostCount
                            self.sTrajectory(idxI).lostCount = 1;

                            shouldTrackID(idxI) = false;
                            shouldAddNewID(idxJ) = false;
                        end

                    end
                end

                %% Information Integration - lost trast
                idxLost = find(shouldTrackID);
                for id = 1:length(idxLost)
                    idx = idxLost(id);
                    % history location----------------------------
                    self.sTrajectory(idxI).history = ...
                        circshift(self.sTrajectory(idxI).history, -1);
                    self.sTrajectory(idxI).history(end, :) = ...
                        [NaN, NaN];
                    self.sTrajectory(idx).lostCount = ...
                        self.sTrajectory(idx).lostCount + 1;
                    self.sTrajectory(idx).direction = NaN;
                    self.sTrajectory(idx).velocity = NaN;
                end

                %% Information Integration - wipe off
                id = 1;
                while id < length(self.sTrajectory)
                    if self.sTrajectory(id).lostCount > ...
                            self.lostThreshold
                        self.sTrajectory(id) = [];
                    else
                        id = id + 1;
                    end
                    
                end

            end


            %% New Trajectory
            listNewID = find(shouldAddNewID);
            for IdNew = listNewID'
                % location
                if isempty(self.sTrajectory)
                     self.sTrajectory = self.initSturct;
                else
                    self.sTrajectory(end+1) = self.initSturct;
                end

                % location------------------------------------
                self.sTrajectory(end).location = newIndex(IdNew,:);
                % history location----------------------------
                self.sTrajectory(end).history = ...
                    circshift(self.sTrajectory(end).history, -1);
                self.sTrajectory(end).history(end, :) = ...
                    self.sTrajectory(end).location;
            end

            %% Direction
            for id = 1:length(self.sTrajectory)
                idX = self.sTrajectory(id).location(1);
                idY = self.sTrajectory(id).location(2);
                if ~isnan(self.sTrajectory(id).direction) ...
                        && ~isnan(idX) && ~isnan(idY)                    
                    mBodyDirection(idX, idY) = ...
                        self.sTrajectory(id).direction;
                end
            end

            %%
            self.Opt = {mBodyResponse, mBodyDirection};

        end

        function record_trajectory_id(self)
            for idxL = 1:length(self.sTrajectory)
                self.sTrajectory(idxL).oldId = idxL;
            end
        end

    end

end
