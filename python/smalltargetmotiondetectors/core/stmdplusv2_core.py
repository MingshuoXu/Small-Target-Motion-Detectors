import numpy as np

from .base_core import BaseCore


class MushroomBody(BaseCore):
    # MushroomBody class for STMDPlus

    def __init__(self):
        # Constructor method
        # Initializes the MushroomBody object
        super().__init__()

        self.maxLenRecordContrast = 100  # Length of Record
        self.SDThres = 5  # Threshold of standard deviation
        self.contrastRecord = []  # trackInfo data

    def init_config(self):
        # Initialization method
        # Initializes the non-maximum suppression
        pass

    def process(self, sTrajectory, lobulaOpt, contrastOpt):
        # Processing method
        # Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt
        mushroomBodyOpt = lobulaOpt

        if not sTrajectory:
            self.contrastRecord = []
            return np.zeros_like(lobulaOpt)

        lenT = len(sTrajectory)
        newContrastRecord = [None] * lenT
        numContrast = len(contrastOpt)

        for newId in range(lenT):
            oldId = sTrajectory[newId]['oldId']

            # lost tract
            if sTrajectory[newId]['lostCount'] > 1:
                newContrastRecord[newId] = self.contrastRecord[oldId]
                continue

            # contrast of new index
            xNew, yNew = sTrajectory[newId]['location']

            nowContrast = np.zeros((numContrast, 1))
            for idCont in range(numContrast):
                nowContrast[idCont, 0] = contrastOpt[idCont][xNew, yNew]

            if np.isnan(oldId):
                # new response
                newContrastRecord[newId] = nowContrast
            else:
                if len(self.contrastRecord[oldId]) >= self.maxLenRecordContrast:
                    newContrast_newId = np.roll(self.contrastRecord[oldId], -1, axis=1)
                    newContrast_newId[:, -1] = nowContrast[:, 0]
                    newContrastRecord[newId] = newContrast_newId
                else:
                    newContrastRecord[newId] = np.hstack((self.contrastRecord[oldId], nowContrast))

                # Small Target Discrimination
                if np.max(np.std(newContrastRecord[newId], axis=1)) < self.SDThres:
                    mushroomBodyOpt[xNew, yNew] = 0

        self.contrastRecord = newContrastRecord
        self.Opt = mushroomBodyOpt
        return mushroomBodyOpt