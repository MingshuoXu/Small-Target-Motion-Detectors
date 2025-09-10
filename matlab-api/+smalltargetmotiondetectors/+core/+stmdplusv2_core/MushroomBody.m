classdef MushroomBody < smalltargetmotiondetectors.core.stmdplus_core.MushroomBody
    % MushroomBody class for STMDPlus
    
    methods
        function self = MushroomBody()
            % Constructor method
            % Initializes the MushroomBody object
            self@smalltargetmotiondetectors.core.stmdplus_core.MushroomBody();
        end
        
        function init_config(self)
            % Initialization method
            % Initializes the non-maximum suppression
            self.init_config@smalltargetmotiondetectors.core.stmdplus_core.MushroomBody();
        end
        
        function mushroomBodyOpt = process(self, lobulaOpt, contrastOpt)
            % Processing method
            % Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt
            import smalltargetmotiondetectors.util.compute.*;
            
            nmsLobulaOpt = self.hNMS.nms(lobulaOpt);
            
            mushroomBodyOpt = lobulaOpt .* (nmsLobulaOpt > 0);
            
            maxNumber = max(nmsLobulaOpt(:));
            
            if maxNumber <= 0
                self.trackID = [];
                self.trackInfo = {};
                self.Opt = mushroomBodyOpt;
                return;
            end
            
            [idX, idY] = find(nmsLobulaOpt > 0);
            newID = [idX, idY];
            
            shouldTrackID = true(size(self.trackID,1 ), 1) & ~isempty(self.trackID);
            shouldAddNewID = true(size(newID, 1), 1);
            numContrast = length(contrastOpt);
            
            if ~isempty(self.trackID)
                if self.hasFunPdist2
                    DD = pdist2(self.trackID, newID);
                else
                    DD = compute_pdist2(self.trackID, newID);
                end
                D1 = min(DD, [], 2);
                
                for idxI = 1:length(D1)
                    if D1(idxI) <= self.DBSCANDist
                        [~, idxJ] = min(DD(idxI, :));
                        if shouldAddNewID(idxJ)
                            self.trackID(idxI, :) = newID(idxJ, :);
                            nowContrast = arrayfun(@(idCont) contrastOpt{idCont}(newID(idxJ, 1), newID(idxJ, 2)), 1:numContrast);
                            self.trackInfo{idxI} = [self.trackInfo{idxI}, nowContrast];
                            shouldTrackID(idxI) = false;
                            shouldAddNewID(idxJ) = false;
                        end
                    end
                end
                
                self.trackID(shouldTrackID, :) = [];
                self.trackInfo(shouldTrackID) = [];
            end
            
            oldTractNum = length(self.trackInfo);
            
            isNew = find(shouldAddNewID);
            for kk = isNew'
                if isempty(self.trackID)
                    self.trackID = newID(kk, :);
                else
                    self.trackID = [self.trackID; newID(kk, :)];
                end
                nowContrast = arrayfun(@(idCont) contrastOpt{idCont}(newID(kk, 1), newID(kk, 2)), 1:numContrast);
                self.trackInfo{end + 1} = nowContrast;
            end
            
            for idx = 1:oldTractNum
                if max(std(self.trackInfo{idx}, [], 2)) < self.SDThres
                    idX = self.trackID(idx, 1);
                    idY = self.trackID(idx, 2);
                    mushroomBodyOpt(idX, idY) = 0;
                end
                
                if size(self.trackInfo{idx}, 2) > self.lenDBSCAN
                    self.trackInfo{idx} = self.trackInfo{idx}(:, 2:end);
                end
            end
            
            self.Opt = mushroomBodyOpt;
        end
    end
end
