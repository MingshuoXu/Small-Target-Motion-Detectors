import warnings

from .backbone import DSTMDBackbone
from ..core import stmdplus_core, apgstmd_core
from ..util.compute_module import compute_response, compute_direction


class STMDPlus(DSTMDBackbone):
    """ STMDPlus: A facilitated model based on DSTMD with an additional contrast pathway.

    Ref:
        * Wang H, Peng J, Zheng X, et al. "A robust visual system for small target motion detection against cluttered moving backgrounds." IEEE Transactions on Neural Networks and Learning Systems, 2019, 31(3): 839-853.

    Description:
        The STMDPlus model builds upon the DSTMD architecture, enhancing target detection accuracy in cluttered moving backgrounds by introducing a contrast pathway. This pathway provides a complementary processing mechanism to improve robustness against dynamic noise and varying background contrasts. The model parameters align with those specified in the reference.

    Parameters:
        Retina:
            - sigma1: Controls the standard deviation for Gaussian blur applied in the retina layer, reducing noise and emphasizing potential targets. (Eq. 1)
        
        Lamina:
            - n1, tau1: Order and time constant of the first gamma bandpass filter in the lamina. (Eq. 3)
            - n2, tau2: Order and time constant of the second gamma bandpass filter.
        
        Medulla:
            - n3, tau3: Order and time constant of gamma delay in the Mi1 pathway. (Eq. 11)
            - n4, tau4: Order and time constant of gamma delay in the Tm1 pathway. (Eq. 11)
            - n5, tau5: Order and time constant for another gamma delay variant in the Tm1 pathway. (Eq. 11)
        
        Lobula:
            - alpha1: Parameter modulating signal strength for the lobula, contributing to directional selectivity. (Eq. 10)
            - A, B: Lateral inhibition coefficients. (Eq. 13)
            - e, rho, sigma4, sigma5: Lateral inhibition coefficients. (Eq. 14)

        Contrast Pathway:
            - eta: Parameter adjusting the contrast pathway’s influence. (Eq. 15)
            - alpha2: Signal modulation factor within the contrast pathway. (Eq. 17)
    """


    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'retina.sigma', # Eq. (1)
        # lamina
        'n1'        : 'lamina.order1',  # Eq. (3)
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order2',
        'tau2'      : 'lamina.tau2',
        # medulla
        'n3'        : 'medulla.mi1_para4.order',  # Eq. (11)
        'tau3'      : 'medulla.mi1_para4.tau',
        'n4'        : 'medulla.tm1_para5.order',
        'tau4'      : 'medulla.tm1_para5.tau',
        'n5'        : 'medulla.tm1_para6.order',
        'tau5'      : 'medulla.tm1_para6.tau',
        # lobula
        'alpha1'    : 'lobula.alpha1',  # Eq. (10)
        'A'         : 'lobula.hLateralInhi.A',  # Eq. (13)
        'B'         : 'lobula.hLateralInhi.B', 
        'e'         : 'lobula.hLateralInhi.e',  # Eq. (14)
        'rho'       : 'lobula.hLateralInhi.rho', 
        'sigma4'    : 'lobula.hLateralInhi.sigma1', 
        'sigma5'    : 'lobula.hLateralInhi.sigma2', 
        # Contrast pathway
        'eta'       : 'contrast_pathway.eta', # Eq. (15)
        'alpha2'    : 'contrast_pathway.alpha2', # Eq. (17)
        } 
    
    def __init__(self):
        """ Constructor method """
        super().__init__()

        # Initialize contrast pathway and mushroom body components
        self.contrast_pathway = stmdplus_core.ContrastPathway()
        self.mushroom_body = stmdplus_core.MushroomBody()

    def forward(self, iptMatrix):
        """ Defines the structure of the STMDPlus model. """      

        # A. Ommatidia (Retina)
        retina_output = self.retina.forward(iptMatrix)

        # B. Motion Pathway (Lamina, Medulla, Lobula)
        lamina_output = self.lamina.forward(retina_output)
        medulla_tm3_output, medulla_mi1_p4_output, medulla_tm1_p5_output, medulla_tm1_p6_output = \
            self.medulla.forward(lamina_output)
        lobula_output = self.lobula.forward(medulla_tm3_output, medulla_mi1_p4_output, 
                                            medulla_tm1_p5_output, medulla_tm1_p6_output)

        # C. Contrast Pathway
        contrast_output = self.contrast_pathway.forward(retina_output)

        # D. Mushroom Body
        mushroom_body_output = self.mushroom_body.forward(
            lobula_output, contrast_output)

        # Compute response and direction
        self.model_output['response'] = compute_response(mushroom_body_output)
        self.model_output['direction'] = compute_direction(mushroom_body_output)

        return self.model_output


class ApgSTMD(STMDPlus):
    """ ApgSTMD: Attention-Prediction-guided Small Target Motion Detector

    Ref: 
        * Wang H, Zhao J, Wang H, et al. Attention and prediction-guided motion detection for low-contrast small moving targets[J]. IEEE Transactions on Cybernetics, 2022, 53(10): 6340-6352.

    Description:
        The ApgSTMD model extends the STMDPlus model by introducing attention and prediction pathways to improve target detection.

    Parameters:            
        Retina:
            - sigma1: Standard deviation of the Gaussian blur applied in the retina layer to pre-forward input images by smoothing, reducing background noise. (Eq. 2)
        
        Lamina:
            - n1, tau1: Order and time constant of the first gamma bandpass filter in the lamina. (Eq. 6)
            - n2, tau2: Order and time constant of the second gamma bandpass filter. (Eq. 6)
        
        Medulla:
            - n3, tau3: Order and time constant in the Mi1 pathway. (Eq. 14)
            - n4, tau4: Order and time constant in the Tm1 pathway. (Eq. 14)
            - n5, tau5: Additional delay component in Tm1. (Eq. 14)
        
        Lobula:
            - gamma: Signal modulation parameter in the lobula, aiding in selective attention towards targets by enhancing certain spatial patterns. (Eq. 13)
            - A, B: Coefficients for lateral inhibition, reducing background clutter through inhibition of surrounding non-target signals. (Eq. 15)
            - e, rho, sigma4, sigma5: Parameters controlling non-linear inhibition effects. (Eq. 16)
        
        Attention Pathway:
            - eta_list: A list of attentional weights adjusting model responsiveness to spatial regions with potential target information. (Eq. 3)
            - theta_list: (Eq. 3)
        
        Prediction Pathway:
            - zeta, eta: Parameters for prediction kernels. (Eq. 20)
            - kappa: Parameter adjusting the weighting for predicted versus observed signals, balancing real-time data with anticipated target trajectories. (Eq. 23)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'retina.sigma', # Eq. (2)
        # lamina
        'n1'        : 'lamina.order1',  # Eq. (6)
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order2',
        'tau2'      : 'lamina.tau2',
        # medulla
        'n3'        : 'medulla.mi1_para4.order',  # Eq. (14)
        'tau3'      : 'medulla.mi1_para4.tau',
        'n4'        : 'medulla.tm1_para5.order',
        'tau4'      : 'medulla.tm1_para5.tau',
        'n5'        : 'medulla.tm1_para6.order',
        'tau5'      : 'medulla.tm1_para6.tau',
        # lobula
        'gamma'    : 'lobula.alpha1',  # Eq. (13)
        'A'         : 'lobula.hLateralInhi.A',  # Eq. (15)
        'B'         : 'lobula.hLateralInhi.B', 
        'e'         : 'lobula.hLateralInhi.e',  # Eq. (16)
        'rho'       : 'lobula.hLateralInhi.rho', 
        'sigma4'    : 'lobula.hLateralInhi.sigma1', 
        'sigma5'    : 'lobula.hLateralInhi.sigma2', 
        # Attention Pathway
        'zeta_list'  : 'attention_pathway.zeta_list', # Eq. (3)
        'theta_list': 'attention_pathway.theta_list',
        # Prediction Pathway
        'zeta'      : 'prediction_pathway.zeta', # Eq. (20)
        'eta'       : 'prediction_pathway.eta', # Eq. (20)
        'kappa'     : 'prediction_pathway.kappa', # Eq. (23)
        } 

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()

        # Initialize attention pathway and prediction pathway components
        self.attention_pathway = apgstmd_core.AttentionModule()
        self.prediction_pathway = apgstmd_core.PredictionModule()

        # Set properties of Lobula's LateralInhibition module
        self.lobula.hLateralInhi.B = 3.5
        self.lobula.hLateralInhi.sigma1 = 1.25
        self.lobula.hLateralInhi.sigma2 = 2.5
        self.lobula.hLateralInhi.e = 1.2

        self.predictionMap = None

    def forward(self, x):
        """ Defines the structure of the ApgSTMD model. """

        # Preprocessing Module
        retina_output = self.retina(x)

        # Attention Module
        attention_output = self.attention_pathway(
            retina_output, self.predictionMap)

        # STMD-based Neural Network
        lamina_output = self.lamina(attention_output)
        medulla_tm3_output, medulla_mi1_p4_output, medulla_tm1_p5_output, medulla_tm1_p6_output = \
            self.medulla(lamina_output)

        lobula_output = self.lobula(medulla_tm3_output, medulla_mi1_p4_output, 
                                    medulla_tm1_p5_output, medulla_tm1_p6_output)

        # STMDPlus
        contrast_output = self.contrast_pathway(retina_output)
        mushroom_body_output = self.mushroom_body(
            lobula_output, contrast_output)

        # Prediction Module
        #   prediction_output is the facilitated STMD output Q(x; y; t; theta) in Eq. (23)
        prediction_output, self.predictionMap = self.prediction_pathway(mushroom_body_output)

        # Compute response and direction
        self.model_output['response'] = compute_response(prediction_output)
        self.model_output['direction'] = compute_direction(prediction_output)

        return self.model_output







