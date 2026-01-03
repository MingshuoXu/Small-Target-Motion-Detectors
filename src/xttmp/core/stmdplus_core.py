from cv2 import filter2D, BORDER_CONSTANT
import numpy as np
from scipy.spatial.distance import cdist

from .base_core import BaseCore
from ..util.create_kernel import create_T1_kernels
from ..util.compute_module import AreaNMS
from ..util.compute_module import compute_response

class ContrastPathway(BaseCore):
    """ContrastPathway class for ApgSTMD."""

    def __init__(self):
        """Constructor method."""
        # Initializes the ContrastPathway object
        super().__init__()
        self.theta = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
        self.alpha2 = 1.5
        self.eta = 3
        self.sizeT1 = 11
        self.T1Kernel = None

    def init_config(self):
        """Initialization method."""
        # Initializes the T1Kernel
        self.T1Kernel = create_T1_kernels(len(self.theta), self.alpha2, self.eta, self.sizeT1)

    def process(self, retinaOpt):
        """Processing method."""
        # Processes the input retinaOpt to generate contrastOpt
        lenKernel = len(self.theta)
        dictContrastOpt = {}
        for idx in range(lenKernel):
            dictContrastOpt[idx] = filter2D(retinaOpt, -1, self.T1Kernel[idx], borderType=BORDER_CONSTANT)
        self.Opt = dictContrastOpt
        return dictContrastOpt
    

class MushroomBody(BaseCore):
    # MushroomBody class for STMDPlus

    def __init__(self):
        # Constructor method
        # Initializes the MushroomBody object
        super().__init__()

        self.paraNMS = {
            'maxRegionSize': 5,
            'method': 'sort'
        }  # Parameters for non-maximum suppression

        self.DBSCANDist = 5  # Spatial distance for clustering
        self.lenDBSCAN = 100  # Length of clustering trajectory
        self.SDThres = 5  # Threshold of standard deviation
        self.T1Kernel = None  # T1 kernel
        self.hNMS = None  # object's handle of non-maximum suppression
        self.trackID = None  # trackInfo index
        self.trackInfo = []  # trackInfo data

    def init_config(self):
        # Initialization method
        # Initializes the non-maximum suppression
        self.hNMS = AreaNMS(self.paraNMS['maxRegionSize'], self.paraNMS['method'])


    def process(self, lobulaOpt, contrastOpt):
        # Processing method
        # Processes the input lobulaOpt and contrastOpt to generate mushroomBodyOpt

        maxLobulaOpt = compute_response(lobulaOpt)
        nmsLobulaOpt = self.hNMS.nms(maxLobulaOpt)

        numDirection = len(lobulaOpt)
        mushroomBodyOpt = [None] * numDirection
        for idxI in range(numDirection):
            mushroomBodyOpt[idxI] = lobulaOpt[idxI] * np.logical_not(nmsLobulaOpt)

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
                for idxDirection in range(numDirection):
                    idX = self.trackID[idx, 0]
                    idY = self.trackID[idx, 1]
                    mushroomBodyOpt[idxDirection][idX, idY] = 0

            if self.trackInfo[idx].shape[1] > self.lenDBSCAN:
                self.trackInfo[idx] = self.trackInfo[idx][:, 1:]

        self.Opt = mushroomBodyOpt
        return mushroomBodyOpt

