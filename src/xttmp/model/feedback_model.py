from copy import deepcopy

import torch

from .backbone import ESTMDBackbone, FracSTMD
from ..core import feedbackstmd_core, fstmd_core, stfeedbackstmd_core


class FeedbackSTMD(ESTMDBackbone):
    """ FeedbackSTMD: Small Target Motion Detector with feedback pathway in lobula

    Ref: 
        * Wang H, Wang H, Zhao J, et al. A time-delay feedback neural network for discriminating small, fast-moving targets in complex dynamic environments[J]. IEEE Transactions on Neural Networks and Learning Systems, 2021, 34(1): 316-330.

    Description:
        The FeedbackSTMD model is an enhancement of the ESTMD backbone with a feedback pathway integrated in the lobula. 

    Parameters:
        Retina:
            - sigma1: Standard deviation of the Gaussian blur in the retina. (Eq. 2)

        Lamina:
            - n1, tau1: Order and time constant for the first gamma bandpass filter in the lamina. (Eq. 3)
            - n2, tau2: Order and time constant for the second gamma filter.

        Medulla:
            - n3, tau3: Tuple containing orders and time constants for Tm1 and Mi1 pathways, both crucial in delaying signals to provide temporal integration for motion detection. (Eq. 7)
        
        Lobula:
            - alpha: Feedback constant in the lobula. (Eq. 9)
            - n4, tau4: Order and time constant in the lobula’s gamma delay, supporting time-delay feedback that strengthens temporal coherence in target tracking.
            - eta: Parameter in the Gaussian kernel. (Eq. 10)
            - A, B: Lateral inhibition coefficients. (Eq. 15)
            - e, rho, sigma2, sigma3: Parameters that modulate nonlinear inhibition. (Eq. 16)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'retina.sigma', # Eq. (2)
        # lamina
        'n1'        : 'lamina.order1', # Eq. (3)
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order1',
        'tau2'      : 'lamina.tau1',
        # medulla
        'n3'        : ('medulla.tm1.order', 'medulla.mi1.order'), # Eq. (7)
        'tau3'      : ('medulla.tm1.tau', 'medulla.mi1.tau'), 
        # lobula
        'alpha'     : 'lobula.alpha', # Eq. (9)
        'n4'        : 'lobula.gamma_delay.order',
        'tau4'      : 'lobula.gamma_delay.tau', 
        'eta'       : 'lobula.sigma', # Eq. (10)
        'A'         : 'lobula.spatial_inhibition.A', # Eq. (15)
        'B'         : 'lobula.spatial_inhibition.B',
        'e'         : 'lobula.spatial_inhibition.e', # Eq. (16)
        'rho'       : 'lobula.spatial_inhibition.rho',
        'sigma2'    : 'lobula.spatial_inhibition.sigma1',
        'sigma3'    : 'lobula.spatial_inhibition.sigma2',
        }
    
    def __init__(self):
        """
        FeedbackSTMD Constructor method
        Initializes an instance of the FeedbackSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Customize Lobula component
        self.lobula = feedbackstmd_core.Lobula()
        
        # Customize Lamina's GammaBankPassFilter properties
        self.lamina.order1 = 4
        self.lamina.tau1 = 8
        self.lamina.order2 = 16
        self.lamina.tau2 = 32

        # Customize Medulla's Tm1 component properties
        self.medulla.tm1.order = 9
        self.medulla.tm1.tau = 45

    def forward(self, x):
        """ MODEL_STRUCTURE Method

        Defines the structure of the FeedbackSTMD model.
        """
        # Process input matrix through model components
        retina_output = self.retina(x)
        lamina_output = self.lamina(retina_output)
        medulla_ON, medulla_OFF = self.medulla(lamina_output)
        self.model_output['response'] = self.lobula(medulla_ON, medulla_OFF)

        return self.model_output


class FSTMD(ESTMDBackbone):
    """ FSTMD: Small Target Motion Detector with feedback loop between lobula and lamina

    Ref: 
        * Ling J, Wang H, Xu M, et al. Mathematical study of neural feedback roles in small target motion detection[J]. Frontiers in Neurorobotics, 2022, 16: 984430.

    Description:
        The FSTMD model introduces a feedback loop between the lobula and lamina layers.

    Parameters:
        Retina:
            - sigma1: Standard deviation of the Gaussian blur in the retina, pre-processing the visual input by smoothing noise and high-frequency signals. (Eq. 1)

        Lamina:
            - n1, tau1: Order and time constant for the first gamma bandpass filter in the lamina. (Eq. 6)
            - n2, tau2: Order and time constant for the second gamma filter.

        Medulla:
            - n3, tau3: Tuple containing the order and time constants for the gamma delay functions in Tm1 and Mi1 pathways, which provide temporal integration critical for target motion analysis. (Eq. 9)

        Lobula:
            - e, rho, sigma2, sigma3: Parameter controlling the strength of lateral inhibition in the lobula. (Eq. 13)

        Feedback Pathway:
            - n4, tau4: Order and time constant for the gamma delay in the feedback pathway, adjusting temporal coherence to align with the dynamics of target motion. (Eq. 4)
            - a: Feedback constant, regulating the strength of the feedback signal from the lobula to the lamina, balancing sensitivity and stability in target detection by dynamically adjusting the lamina’s response to target motion. (Eq. 4)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'retina.sigma', # Eq. (1)
        # lamina
        'n1'        : 'lamina.order1', # Eq. (6)
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order2',
        'tau2'      : 'lamina.tau2',
        # medulla
        'n3'        : ('medulla.tm1.order', 'medulla.mi1.order'), # Eq. (9)
        'tau3'      : ('medulla.tm1.tau', 'medulla.mi1.tau'), 
        # lobula
        'e'         : 'lobula.spatial_inhibition.e', # Eq. (13)
        'rho'       : 'lobula.spatial_inhibition.rho',
        'sigma2'    : 'lobula.spatial_inhibition.sigma1',
        'sigma3'    : 'lobula.spatial_inhibition.sigma2',
        # feedback pathway
        'n4'        : 'feedback_pathway.order', # Eq. (4)
        'tau4'      : 'feedback_pathway.tau', 
        'a'         : 'feedback_pathway.feedback_coefficient', # Eq. (4)
        }
    
    def __init__(self):
        """ FSTMD Constructor method

        Initializes an instance of the FSTMD class.
        """
        # Call superclass constructor
        super().__init__()
        
        # Initialize feedback pathway component
        self.feedback_pathway = fstmd_core.FeedbackPathway()

        self.maxIterationNum = 10
        self.iterationThreshold = 1e-3

        self.set_para(n3=5)

    def forward(self, x):
        """ MODEL_STRUCTURE Method

        Defines the structure of the FSTMD model.
        """
        last_feedback_output = torch.ones_like(x)
        self.feedback_output = torch.zeros_like(x)

        # Retina layer
        retina_output = self.retina(x)

        # Feedback loop
        iteration_count = 1
        self.set_loop_state(False)
        while iteration_count < self.maxIterationNum and torch.max(
                torch.abs(self.feedback_output - last_feedback_output)) > self.iterationThreshold:
            last_feedback_output = self.feedback_output.clone()

            # Execute feedback loop
            lamina_output = self.lamina(retina_output + self.feedback_output)
            medulla_ON, medulla_OFF = self.medulla(lamina_output)
            lobula_output, correlation_output = self.lobula(medulla_ON, medulla_OFF)
            self.feedback_output = self.feedback_pathway(correlation_output)

            iteration_count += 1
            self.set_loop_state(True)

        # Set model response
        self.model_output['response'] = lobula_output

        return self.model_output

    def set_loop_state(self, state):
        """ Sets the loop state of certain components. """
        # Disable circshift for certain components
        self.lamina.in_loop = state
        self.medulla.tm1.in_loop = state
        self.feedback_pathway.in_loop = state


class STFeedbackSTMD(ESTMDBackbone):
    """ STFeedbackSTMD - Small Target Motion Detection With Spatio-Temporal Feedback

    # !!! Notice: This model has not been reproduced yet
    
    Ref: 
        [1] Wang H, Zhong Z, Lei F, et al. Bio-Inspired Small Target Motion Detection With Spatio-Temporal Feedback in Natural Scenes[J]. IEEE Transactions on Image Processing, 2023.

    
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {} 

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()       

        # Customize Medulla and Lobula component
        self.medulla = stfeedbackstmd_core.Medulla()
        self.lobula = stfeedbackstmd_core.Lobula()


class FracSTMD_F(FracSTMD):

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = deepcopy(FracSTMD._FracSTMD__paraMappingList)
    __paraMappingList.update({
        # lobula
        'beta'      : 'lobula.alpha', # Eq. (9)
        'n4'        : 'lobula.hGammaDelay.order',
        'tau4'      : 'lobula.hGammaDelay.tau', 
        'eta'       : 'lobula.paraGaussKernel[\'eta\']', # Eq. (10)
    })

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()       

        # Customize Lamina component to include fractional differentiation
        self.lobula = feedbackstmd_core.Lobula()

    def forward(self, iptMatrix):
        """ MODEL_STRUCTURE Method

        Defines the structure of the FeedbackSTMD model.
        """
        # Process input matrix through model components
        self.retinaOpt = self.retina.forward(iptMatrix)
        lamina_output = self.lamina.forward(self.retinaOpt)
        self.medulla.forward(lamina_output)
        self.medullaOpt = self.medulla.Opt
        lobula_output = self.lobula.forward(self.medullaOpt[0], self.medullaOpt[1])

        # Set model response
        self.model_output['response'] = lobula_output







