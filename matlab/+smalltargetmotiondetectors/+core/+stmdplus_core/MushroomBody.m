classdef MushroomBody < smalltargetmotiondetectors.core.BaseCore
    % MushroomBody class for STMDPlus

    properties
        paraNMS = struct( ...
            'maxRegionSize', 15, ...
            'method', 'sort'); % Parameters for non-maximum suppression

        detectThreshold = 0.01; % Response threshold for clustering
        DBSCANDist = 5; % Spatial distance for clustering
        lenDBSCAN = 100; % Length of clustering trajectory
        SDThreshold = 15; % Threshold of standard deviation
    end

    properties(Hidden)
        T1Kernel; % T1 kernel
        hNMS; % Non-maximum suppression
        trackIdx; % Track index
        track = {}; % Track data
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

        end
        
        function mushroomBodyOpt = process(self, lobulaOpt, contrastOpt)
            % Processing method
            % Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt
            
            import smalltargetmotiondetectors.tool.compute.*;

            maxLobulaOpt = compute_response(lobulaOpt);
            nmsLobulaOpt = self.hNMS.nms(maxLobulaOpt);

            numDirection = length(lobulaOpt);
            mushroomBodyOpt = cell(numDirection, 1);
            for idx = 1 : numDirection
                mushroomBodyOpt{idx} = ...
                    lobulaOpt{idx} .* logical(nmsLobulaOpt);
            end
            max_ = max(nmsLobulaOpt, [], 'all');

            if max_ > 0
                if max_ ~= 1
                    nornroomOpt = nmsLobulaOpt ./ max_;
                else
                    nornroomOpt = nmsLobulaOpt;
                end
            else
                self.trackIdx = [];
                self.track = {};
                return;
            end

            [idX, idY] = find(nornroomOpt > self.detectThreshold);
            indexXY = [idX, idY];

            %% Information Integration - join
            stateIdx = false(length(idX), 1);
            stateTRIdx = false(size(self.trackIdx,1), 1);
            
            if ~isempty(self.trackIdx)
                DD = pdist2(self.trackIdx, indexXY);

                [D1, ind1] = min(DD,[],2);

                for ii = 1:length(D1)
                    if D1(ii) <= self.DBSCANDist
                        j = ind1(ii);
                        self.trackIdx(ii,:) = indexXY(j,:);
                        stateTRIdx(ii) = true;
                        stateIdx(j) = true;
                        self.track{ii} = [
                            self.track{ii}, ...
                            squeeze(contrastOpt(indexXY(j,1),indexXY(j,2),:))...
                            ];
                    end
                end
                jj_ = find(~stateTRIdx);
                self.trackIdx(jj_,:) = [];
                self.track(jj_) = [];

            end

            hasTractNum = length(self.track);

            %% Information Integration - wipe off
            kk_ = find(~stateIdx);
            for kk = kk_'
                self.trackIdx(end+1,:) = indexXY(kk,:);
                self.track{end+1} = squeeze( ...
                    contrastOpt(indexXY(kk,1), indexXY(kk,2), :) ) ;
            end

            %% Small Target Discrimination

            for idx = 1:length(hasTractNum)
                
                if max(std(self.track{idx}, 0, 2)) < self.SDThreshold
                    for idxDirection = 1 : numDirection
                        idX = self.trackIdx(idx,1);
                        idY = self.trackIdx(idx,2);
                        mushroomBodyOpt{idxDirection}(idX, idY) = 0;
                    end
                end

                % Limit the length to ensure that the contents are 
                %   within reasonable limits
                if size(self.track{idx},2) > self.lenDBSCAN
                    self.track{idx} = self.track{idx}(:,2:end);
                end
            end
            self.Opt = mushroomBodyOpt;
        end
    end

end
