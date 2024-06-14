import numpy as np

from cv2 import filter2D

from .base_core import BaseCore
from ..util.datarecord import CircularList
from ..util.compute_module import compute_circularlist_conv
from .math_operator import SurroundInhibition


class Medulla(BaseCore):
    def __init__(self):
        super().__init__()
        self.lenTemporalKernel = 15
        self.sizeSpacialKernel = [8, 16]
        self.cp = 15  # a parameter to adjust the spacialOn and spacialOff
        self.theta = np.pi

        self.spacialOnKernel = None
        self.spacialOffKernel = None
        self.temporalOnKernel = None
        self.temporalOffKernel = None

        self.cellSpatialOpt = CircularList()
        self.cellMedullaIpt = CircularList()

        self.init_config()

    def init_config(self):
        # Spacial kernels
        m1 = self.sizeSpacialKernel[0]
        n1 = int(np.ceil(self.sizeSpacialKernel[1] / 2))
        n2 = self.sizeSpacialKernel[1] - n1

        self.spacialOnKernel = np.vstack((np.ones((m1, n1)), np.zeros((m1, n2))))
        self.spacialOffKernel = np.vstack((np.zeros((m1, n1)), -np.ones((m1, n2))))

        # Temporal kernels
        k1 = int(np.ceil(self.lenTemporalKernel / 2))
        k2 = self.lenTemporalKernel - k1

        self.temporalOnKernel = np.ones((k1, 1))
        self.temporalOffKernel = np.vstack((np.zeros((k1, 1)), -np.ones((k2, 1))))

        # Allocate memory
        self.cellSpatialOpt.initLen = self.lenTemporalKernel
        self.cellSpatialOpt.reset()

        self.cellMedullaIpt.initLen = self.lenTemporalKernel
        self.cellMedullaIpt.reset()

    def process(self, medullaIpt):
        ''' Compute spacial part '''
        # SP_ON
        spacialOnOpt = np.maximum(filter2D(medullaIpt, -1, self.spacialOnKernel), 0)
        # SP_OFF
        spacialOffOpt = np.maximum(filter2D(medullaIpt, -1, self.spacialOffKernel), 0)

        # SP
        nowSpacialOpt = self.compute_spacial_correlation(spacialOnOpt, 
                                                         spacialOffOpt, 
                                                         self.cp, 
                                                         self.theta)
        if nowSpacialOpt.max() > 0:
            nowSpacialOpt /= nowSpacialOpt.max()
        # record Spatial output by cell (Parameter_Residual.DLSTMD_SpatialSum)
        self.cellSpatialOpt.record_next(nowSpacialOpt)

        ''' Compute temporal part '''
        # record Medulla input (Lamina output) by cell (Parameter_Residual.DLSTMD_SpatialSum)
        self.cellMedullaIpt.record_next(medullaIpt)

        temporalOnOpt = np.maximum(
            compute_circularlist_conv(self.cellMedullaIpt, self.temporalOnKernel), 
            0)
        temporalOffOpt = np.maximum(
            compute_circularlist_conv(self.cellMedullaIpt, self.temporalOffKernel), 
            0)

        # TP
        # There's no need for half-wave rectification here
        temporalOpt = temporalOnOpt * temporalOffOpt

        # Store the output in Opt property
        self.Opt = {'cellSpatialOpt': self.cellSpatialOpt, 'temporalOpt': temporalOpt}

        return self.cellSpatialOpt, temporalOpt

    @classmethod
    def compute_spacial_correlation(cls, spacialOnOpt, spacialOffOpt, alpha, theta):
        spacialOpt = np.zeros_like(spacialOnOpt)

        dColumn = round(alpha * np.cos(theta))
        if theta <= np.pi:
            dLine = -abs(round(alpha * np.sin(theta)))
        else:
            dLine = abs(round(alpha * np.sin(theta)))

        bw = round(alpha) # bw = round(alpha*sin(pi/2))

        spacialOpt[bw:-bw, bw:-bw] = spacialOnOpt[bw:-bw, bw:-bw] * \
                                    spacialOffOpt[bw+dLine:-bw+dLine, bw+dColumn:-bw+dColumn]

        return spacialOpt


class Lobula(BaseCore):
    def __init__(self):
        super().__init__()
        self.tau = 1  # a parameter to align the spacialOpt and temporalOpt
        self.hSubInhi = SurroundInhibition()
        self.hSubInhi.B = 1

    def init_config(self):
        self.hSubInhi.init_config()

    def process(self, listSpatialOpt, temporalOpt):
        # Perform the correlation operation
        pointer = (listSpatialOpt.pointer - self.tau) % listSpatialOpt.initLen
        spatialOpt = listSpatialOpt[pointer]

        if spatialOpt is not None:
            correlationOutput = spatialOpt * temporalOpt
        else:
            correlationOutput = np.zeros_like(temporalOpt)

        # Apply surround inhibition
        lobulaOpt = np.maximum(self.hSubInhi.process(correlationOutput), 0)

        # Store the output in Opt property
        self.Opt = lobulaOpt
        return lobulaOpt


