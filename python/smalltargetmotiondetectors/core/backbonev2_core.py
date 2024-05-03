import numpy as np

from ..base_core import BaseCore
from .. import SurroundInhibition
from ...util.matrixnms import MatrixNMS
from scipy.spatial.distance import pdist2

class MedullaCell(BaseCore):
    # MedullaCell cell in Medulla layer

    def __init__(self):
        super().__init__()

        self.afterhyperpolarizationV = -50
        self.coeffDecay = 0.8
        self.coeffInhi = 0.5
        self.Voltage = None
        self.lastInput = None

    def init_config(self):
        pass

    def process(self, iptMatrix):
        if self.lastInput is None:
            m, n = iptMatrix.shape
            self.Voltage = np.zeros((m, n))
            self.lastInput = np.zeros((m, n), dtype=bool)

        # Decay
        isDecay = self.lastInput
        self.Voltage[isDecay] = self.coeffDecay * self.Voltage[isDecay]

        # Inhi
        isInhi = ~self.lastInput
        self.Voltage[isInhi] = self.coeffInhi * self.Voltage[isInhi]

        # Accumulation
        self.Voltage += iptMatrix

        # record
        self.lastInput = iptMatrix.astype(bool)

        return self.Voltage


class Medulla(BaseCore):
    """
    Medulla layer of the motion detection system.

    Illustration:

    Lamina   LMC1 (L1)            LMC2 (L2)   
                |          *         |          
    ------------|--------------------|--------- 
    Medulla     |  {ON}    *         |  {OFF}   
                |          *         v          
                |    /-------<-<--- Tm2         
                v    |     *         |          
               Mi1 --|->->----\      |          
                |    v     *   v     v          
                |   Mi9    *  TmOn   |          
                |    |     *   |     |          
                v    v     *   v     v          
               Mi4 <-/     *   \->-> Tm9         
                |          *         |          
    ------------|--------------------|--------- 
    Lobula      |          *         |
                v                    v          

    """

    def __init__(self):
        super().__init__()
        # Initialize components
        self.hTm2 = Tm2()
        self.hTm3 = Tm3()

    def init_config(self):
        # Initialize configurations
        self.hTm2.init_config()
        self.hTm3.init_config()

    def process(self, medullaIpt):
        """
        Process the input through the Medulla layer.

        Args:
        - medullaIpt (array-like): Input to the Medulla layer.

        Returns:
        - tm3Signal (array-like): Output signal from Tm3.
        - tm2Signal (array-like): Output signal from Tm2.
        """
        # Process through Tm2 and Tm3
        tm2Signal = self.hTm2.process(medullaIpt)
        tm3Signal = self.hTm3.process(medullaIpt)

        # Store the output signals
        self.Opt = [tm3Signal, tm2Signal]

        return tm3Signal, tm2Signal


class Tm2(MedullaCell):
    """
    Tm2 cell in the Medulla layer.
    """

    def __init__(self):
        super().__init__()

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, tm2Ipt):
        """
        Processing method.

        Args:
        - tm2Ipt (array-like): Input to the Tm2 cell.

        Returns:
        - tm2Opt (array-like): Output of the Tm2 cell.
        """
        offSignal = tm2Ipt.copy()
        offSignal[offSignal > 0] = 0
        tm2Opt = super().process(offSignal)
        self.Opt = tm2Opt
        return tm2Opt


class Tm3(MedullaCell):
    """
    Tm3 cell in the Medulla layer.
    """

    def __init__(self):
        super().__init__()

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, tm3Ipt):
        """
        Processing method.

        Args:
        - tm3Ipt (array-like): Input to the Tm3 cell.

        Returns:
        - tm3Opt (array-like): Output of the Tm3 cell.
        """
        onSignal = tm3Ipt.copy()
        onSignal[onSignal < 0] = 0
        tm3Opt = super().process(onSignal)
        self.Opt = tm3Opt
        return tm3Opt


class Lobula(BaseCore):
    """
    Lobula layer of the motion detection system.
    """

    def __init__(self):
        super().__init__()
        self.hSubInhi = SurroundInhibition()
        self.hDireCell = DirectionCell()

    def init_config(self):
        """
        Initialization method.
        """
        self.hSubInhi.init_config()
        self.hDireCell.init_config()

    def process(self, varagein):
        """
        Processing method.

        Args:
        - varagein (tuple): Tuple containing the ON and OFF channel signals.

        Returns:
        - varargout (tuple): Tuple containing the processed output(s).
        """
        onSignal, offSignal = varagein

        correlationOutput = onSignal * offSignal

        inhiOpt = self.hSubInhi.process(correlationOutput)

        if len(varagein) == 1:
            varargout = (inhiOpt,)
        elif len(varagein) == 2:
            lobulaOpt, direction = self.hDireCell.process(inhiOpt)
            varargout = (lobulaOpt, direction)
        elif len(varagein) == 3:
            lobulaOpt, direction = self.hDireCell.process(inhiOpt)
            varargout = (lobulaOpt, direction, correlationOutput)

        self.Opt = varargout
        return varargout


class DirectionCell(BaseCore):
    """
    Direction in backbonev2.
    """

    def __init__(self):
        """
        Constructor method.
        """
        super().__init__()
        self.paraNMS = {
            'maxRegionSize': 10,
            'method': 'sort'
        }
        self.detectThreshold = 0.01
        self.DBSCANDist = 5
        self.lenDBSCAN = 10
        self.sTrajectory = []

        self.hNMS = MatrixNMS(self.paraNMS['maxRegionSize'], self.paraNMS['method'])
        self.numResponse = 0
        self.lostThreshold = 8

        self.initSturct = {
            'location': [np.nan, np.nan],
            'oldId': np.nan,
            'history': np.nan * np.ones((6, 2)),
            'direction': np.nan,
            'velocity': np.nan,
            'accuV': np.nan,
            'lostCount': 1
        }

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, lobulaOpt):
        """
        Processing method.

        Args:
        - lobulaOpt (numpy.ndarray): Lobula output.

        Returns:
        - mBodyResponse (numpy.ndarray): Mushroom body response.
        - mBodyDirection (numpy.ndarray): Mushroom body direction.
        """
        mBodyResponse = self.hNMS.nms(lobulaOpt)
        mBodyDirection = np.full_like(mBodyResponse, np.nan)

        self.record_trajectory_id()

        idX, idY = np.where(mBodyResponse > 0)
        self.numResponse = len(idX)

        if self.numResponse == 0:
            self.sTrajectory = []
            return mBodyResponse, mBodyDirection

        newIndex = np.column_stack((idX, idY))

        shouldTrackID = np.ones(len(self.sTrajectory), dtype=bool)
        shouldAddNewID = np.ones(self.numResponse, dtype=bool)

        if self.sTrajectory:
            trajectoryLocation = np.array([traj['location'] for traj in self.sTrajectory])

            DD = pdist2(trajectoryLocation, newIndex)

            ind1 = np.argmin(DD, axis=1)
            D1 = np.min(DD, axis=1)

            for idxI, D1_val in enumerate(D1):
                if D1_val <= self.DBSCANDist:
                    idxJ = ind1[idxI]
                    if shouldAddNewID[idxJ]:
                        self.sTrajectory[idxI]['location'] = newIndex[idxJ]

                        self.sTrajectory[idxI]['history'] = np.roll(self.sTrajectory[idxI]['history'], -1, axis=0)
                        self.sTrajectory[idxI]['history'][-1] = newIndex[idxJ]

                        notNanHist = self.sTrajectory[idxI]['history'][~np.isnan(self.sTrajectory[idxI]['history'][:, 0])]
                        if notNanHist.shape[0] > 1:
                            self.sTrajectory[idxI]['direction'] = get_direction_by_multipoints(notNanHist)

                        self.sTrajectory[idxI]['lostCount'] = 1
                        shouldTrackID[idxI] = False
                        shouldAddNewID[idxJ] = False

            idxLost = np.where(shouldTrackID)[0]
            for idx in idxLost:
                self.sTrajectory[idx]['history'] = np.roll(self.sTrajectory[idx]['history'], -1, axis=0)
                self.sTrajectory[idx]['history'][-1] = [np.nan, np.nan]
                self.sTrajectory[idx]['lostCount'] += 1
                self.sTrajectory[idx]['direction'] = np.nan

            self.sTrajectory = [traj for traj in self.sTrajectory if traj['lostCount'] <= self.lostThreshold]

        listNewID = np.where(shouldAddNewID)[0]
        for IdNew in listNewID:
            self.sTrajectory.append(self.initSturct.copy())
            self.sTrajectory[-1]['location'] = newIndex[IdNew]
            self.sTrajectory[-1]['history'][-1] = newIndex[IdNew]

        for idx, traj in enumerate(self.sTrajectory):
            idX, idY = traj['location']
            if not np.isnan(traj['direction']) and not np.isnan(idX) and not np.isnan(idY):
                mBodyDirection[idX, idY] = traj['direction']

        self.Opt = [mBodyResponse, mBodyDirection]
        return mBodyResponse, mBodyDirection

    def record_trajectory_id(self):
        """
        Record trajectory ID method.
        """
        for idx, traj in enumerate(self.sTrajectory):
            traj['oldId'] = idx


def get_direction_by_multipoints(points):
    """
    get_direction_by_multipoints
    
    ref: https://blog.csdn.net/qwertyu_1234567/article/details/117918602
    """

    differences = np.diff(points, axis=0)
    if np.all(differences == 0):
        theta = np.nan
        return theta

    numPoint = points.shape[0]
    listT = np.arange(1, numPoint + 1)
    listX = points[:, 0]
    listY = points[:, 1]

    sumX = np.sum(listX)
    sumY = np.sum(listY)
    sumT = np.sum(listT)
    sumXT = np.dot(listX, listT)
    SumYT = np.dot(listY, listT)

    veloX = numPoint * sumXT - sumX * sumT
    veloY = numPoint * SumYT - sumY * sumT

    theta = np.arctan2(veloY, veloX) - np.pi / 2

    return theta


def get_multi_direction_opt(modelResponse, modelDirection, numDirection):
    """
    Computes the response for multiple directions based on model response and direction.

    Parameters:
        modelResponse (ndarray): Model response.
        modelDirection (ndarray): Model direction.
        numDirection (int): Number of directions.

    Returns:
        list: List of response for multiple directions.
    """
    directionOutput = []
    directionList = np.arange(0, 2 * np.pi, 2 * np.pi / numDirection)

    notNanId = ~np.isnan(modelDirection)
    nanSub = np.where(notNanId)[0]
    nanInd = np.unravel_index(nanSub, modelDirection.shape)

    for idxDire in range(numDirection):
        cosDire = np.cos(modelDirection[nanInd] - directionList[idxDire])
        cosDire[cosDire < 0] = 0

        Opt = np.zeros_like(modelResponse)
        Opt[nanInd] = modelResponse[nanInd] * cosDire
        directionOutput.append(Opt)

    return directionOutput
