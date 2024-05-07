import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .base_core import BaseCore
from .math_operator import SurroundInhibition, GammaDelay
from . import estmd_backbone 
from ..util.datarecord import CircularList
from ..util.create_kernel import *
from ..util.compute_module import slice_matrix_holding_size

class Medulla(estmd_backbone.Medulla):
    # Medulla layer of the motion detection system

    def __init__(self):
        # Constructor method
        # Initializes the Medulla object
        super().__init__()

        self.hPara5Mi1 = None
        self.hPara5Tm1 = None
        self.cellTm1Ipt = None

    def init_config(self):
        # Initialization method
        # This method initializes the Medulla layer components
        super().init_config()

        self.hTm1 = GammaDelay(5, 25)
        self.hPara5Mi1 = GammaDelay(25, 30)
        self.hPara5Tm1 = GammaDelay(25, 30)

        self.cellTm1Ipt = CircularList()

        self.hTm1.init_config(False)
        self.hPara5Mi1.init_config()
        self.hPara5Tm1.init_config(False)

        if not self.cellTm1Ipt.len:
            self.cellTm1Ipt.len = max(self.hPara5Mi1.lenKernel, self.hPara5Tm1.lenKernel)

        self.cellTm1Ipt.init_config()

    def process(self, MedullaIpt):
        # Processing method
        # Applies processing to the input and returns the output
        # Process Tm2 and Tm3 components
        tm2Signal = self.hTm2.process(MedullaIpt)
        tm3Signal = self.hTm3.process(MedullaIpt)

        # Process Tm1 component using output of Tm2
        self.cellTm1Ipt.circrecord(tm3Signal)
        tm1Para3Signal = self.hTm1.process(self.cellTm1Ipt)
        tm1Para5Signal = self.hPara5Tm1.process(self.cellTm1Ipt)

        # Process Mi1 component using output of Tm3
        mi1Para5Signal = self.hPara5Mi1.process(tm3Signal)

        # Store the output signals in Opt property
        varageout = [tm3Signal, tm1Para3Signal, mi1Para5Signal, tm2Signal, tm1Para5Signal, self.hPara5Mi1.tau]
        self.Opt = varageout
        return varageout


class Lobula(BaseCore):
    # Lobula layer of the motion detection system

    def __init__(self):
        # Constructor method
        # Initializes the Lobula object
        super().__init__()

        self.hSTMD = None
        self.hLPTC = None

    def init_config(self):
        # Initialization method
        # This method initializes the Lobula layer component
        self.hSTMD = Stmdcell()
        self.hLPTC = Lptcell()
        self.hSTMD.init_config()
        self.hLPTC.init_config(self.hSTMD.hGammaDelay.lenKernel)

    def process(self, varagein):
        # Processing method
        # Performs a correlation operation on the ON and OFF channels
        # and then applies surround inhibition

        # Extract ON and OFF channel signals from the input
        tm3Signal, tm1Para3Signal, mi1Para5Signal, tm2Signal, tm1Para5Signal, tau5 = varagein

        psi, fai = self.hLPTC.process(tm3Signal, mi1Para5Signal, tm2Signal, tm1Para5Signal, tau5)

        lobulaOpt = self.hSTMD.process(tm3Signal, tm1Para3Signal, psi, fai)

        # Store the output in Opt property
        self.Opt = lobulaOpt

        return lobulaOpt, []


class Stmdcell(BaseCore):
    # Lobula layer of the motion detection system

    def __init__(self):
        # Constructor method
        # Initializes the Lobula object
        super().__init__()

        self.hSubInhi = None  # SurroundInhibition component
        self.alpha = 0.1  # Parameter alpha
        self.gaussKernel = None  # Gaussian kernel
        self.hGammaDelay = None
        self.cellDPlusE = None
        self.paraGaussKernel = {'size': 3, 'eta': 1.5}

    def init_config(self):
        # Initialization method
        # This method initializes the Lobula layer component
        self.hSubInhi = SurroundInhibition()
        self.hGammaDelay = GammaDelay(6, 12)
        self.cellDPlusE = CircularList()

        self.hSubInhi.init_config()
        self.hGammaDelay.init_config()

        if not self.cellDPlusE.len:
            self.cellDPlusE.len = self.hGammaDelay.lenKernel
        self.cellDPlusE.init_config()

        self.gaussKernel = gaussian_filter(
            np.zeros((self.paraGaussKernel['size'], self.paraGaussKernel['size'])),
            self.paraGaussKernel['eta']
        )

    def process(self, tm3Signal, tm1Signal, faiList, psiList):
        # Processing method
        # Performs temporal convolution, correlation, and surround inhibition
        convnIpt = [None] * self.cellDPlusE.len

        for idxT in range(len(convnIpt)-1, -1, -1):
            point = self.cellDPlusE.point
            if self.cellDPlusE.data[point] is not None:
                fai = faiList[idxT]
                psi = psiList[idxT]
                convnIpt[idxT] = slice_matrix_holding_size(self.cellDPlusE.data[point], psi, fai)

            if point == 0:
                point = self.cellDPlusE.len - 1
            else:
                point -= 1

        feedbackSignal = self.hGammaDelay.process_list(convnIpt)         

        if feedbackSignal is not None:
            feedbackSignal *= self.alpha 
            correlationD = np.maximum(tm3Signal - feedbackSignal, 0) * np.maximum(tm1Signal - feedbackSignal, 0)
        else:
            correlationD = np.maximum(tm3Signal, 0) * np.maximum(tm1Signal, 0)

        correlationE = cv2.filter2D(tm3Signal * tm1Signal, -1, self.gaussKernel)

        lateralInhiSTMDOpt = self.hSubInhi.process(correlationD)

        self.cellDPlusE.circrecord(correlationD + correlationE)

        self.Opt = lateralInhiSTMDOpt

        return lateralInhiSTMDOpt


class Lptcell(BaseCore):
    # Lptcell Lobula Plate Tangential Cell

    def __init__(self):
        # Constructor method
        # Initializes the Lobula object
        super().__init__()

        self.bataList = list(range(2, 19, 2))
        self.thetaList = np.arange(0, 2 * np.pi, np.pi / 4)
        self.velocity = None
        self.tuningCurvef = None

    def init_config(self, lenVelocity):
        self.velocity = np.zeros(lenVelocity)

        lenBataList = len(self.bataList)
        lenThetaList = len(self.thetaList)
        # generate gauss distribution
        gaussianDistribution = np.exp(-0.5 * ((np.arange(-199, 201) - 1) / (100 / 2)) ** 2)
        # normalization
        gaussianDistribution /= np.max(gaussianDistribution)

        self.tuningCurvef = np.zeros((lenBataList, lenThetaList * 100 + 200))
        self.tuningCurvef[0, :300] = gaussianDistribution[100:400]
        self.tuningCurvef[-1, -300:] = gaussianDistribution[:300]
        for id in range(1, lenBataList - 1):
            idRange = slice((id+1) * 100 - 200, (id+1) * 100 + 200)
            self.tuningCurvef[id, idRange] = gaussianDistribution

    def process(self, tm1Signal, tm2Signal, tm3Signal, mi1Signal, tau5):
        lenBataList = len(self.bataList)
        lenThetaList = len(self.thetaList)
        sumLplcOptR = np.zeros((lenBataList, lenThetaList))

        for idBata, bata in enumerate(self.bataList):
            for idTheta, theta in enumerate(self.thetaList):
                shiftX = np.round(bata * np.cos(theta + np.pi / 2)).astype(int)
                shiftY = np.round(bata * np.sin(theta + np.pi / 2)).astype(int)
                shiftMi1Signal = slice_matrix_holding_size(mi1Signal, shiftY, shiftX)
                shiftTm1Signal = slice_matrix_holding_size(tm1Signal, shiftY, shiftX)

                ltlcOpt = tm3Signal * shiftMi1Signal + tm2Signal * shiftTm1Signal

                sumLplcOpt = np.sum(ltlcOpt)

                sumLplcOptR[idBata, idTheta] = sumLplcOpt

        # preferTheta
        firingRate = np.max(sumLplcOptR, axis=1)
        preferTheta = np.argmax(sumLplcOptR, axis=1)
        maxTheta = np.max(preferTheta)

        # background velocity
        self.velocity = np.roll(self.velocity, -1)
        temp = [np.sum((firingRate[i] - self.tuningCurvef[i,:]) ** 2) for i in range(len(firingRate))]
        self.velocity[-1] = np.argmin(temp)

        sumV = np.zeros_like(self.velocity)
        for idV in range(len(self.velocity)):
            sumV[idV] = np.sum(self.velocity[idV:])

        fai = sumV * np.cos(maxTheta)
        psi = sumV * np.sin(maxTheta)

        self.Opt = [fai, psi]

        return fai, psi


