import numpy as np
from scipy.spatial.distance import cdist

from . import stmdplus_core

class MushroomBody(stmdplus_core.MushroomBody):
    # MushroomBody class for STMDPlus

    def __init__(self):
        # Constructor method
        # Initializes the MushroomBody object
        super().__init__()


    def init_config(self):
        # Initialization method
        # Initializes the non-maximum suppression
        super().init_config()

    def process(self, lobulaOpt, contrastOpt):
        # Processing method
        # Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt

        nmsLobulaOpt = self.hNMS.nms(lobulaOpt)

        mushroomBodyOpt = lobulaOpt * (nmsLobulaOpt > 0)

        maxNumber = np.max(nmsLobulaOpt)

        if maxNumber <= 0:
            self.trackID = None
            self.trackInfo = []
            return mushroomBodyOpt

        idX, idY = np.where(nmsLobulaOpt > 0)
        newID = np.column_stack((idX, idY))

        shouldTrackID = np.ones(len(self.trackID), dtype=bool) if self.trackID is not None else np.array([], dtype=bool)
        shouldAddNewID = np.ones(len(idX), dtype=bool)
        numContrast = len(contrastOpt)

        if self.trackID is not None:
            DD = cdist(self.trackID, newID)
            D1 = np.min(DD, axis=1)

            for idxI, d1 in enumerate(D1):
                if d1 <= self.DBSCANDist:
                    idxJ = np.argmin(DD[idxI])
                    if shouldAddNewID[idxJ]:
                        self.trackID[idxI] = newID[idxJ]
                        nowContrast = np.array(
                            [[contrastOpt[idCont][newID[idxJ, 0], newID[idxJ, 1]]] for idCont in range(numContrast)])
                        self.trackInfo[idxI] = np.hstack((self.trackInfo[idxI], nowContrast))
                        shouldTrackID[idxI] = False
                        shouldAddNewID[idxJ] = False

            self.trackID = np.delete(self.trackID, np.where(shouldTrackID), axis=0)
            self.trackInfo = [x for idx, x in enumerate(self.trackInfo) if not shouldTrackID[idx]]

        oldTractNum = len(self.trackInfo)

        isxNew = np.where(shouldAddNewID)[0]
        for kk in isxNew:
            if self.trackID is None:
                self.trackID = newID[kk]
            else:
                self.trackID = np.vstack((self.trackID, newID[kk]))
            nowContrast = np.array(
                [[contrastOpt[idCont][newID[kk, 0], newID[kk, 1]]] for idCont in range(numContrast)])
            self.trackInfo.append(nowContrast)

        for idx in range(oldTractNum):
            if np.max(np.std(self.trackInfo[idx], axis=1)) < self.SDThres:
                idX = self.trackID[idx, 0]
                idY = self.trackID[idx, 1]
                mushroomBodyOpt[idX, idY] = 0

            if self.trackInfo[idx].shape[1] > self.lenDBSCAN:
                self.trackInfo[idx] = self.trackInfo[idx][:, 1:]

        self.Opt = mushroomBodyOpt
        return mushroomBodyOpt