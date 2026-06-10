import warnings

from ..core import haarstmd_core
from .backbone import ESTMDBackbone
from ..util.compute_module import compute_response, compute_direction


class HaarSTMD(ESTMDBackbone):
    ''' HaarSTMD: Advancing small target motion detection in dim light
    
    Ref: 
        [1] Chen H, Sun X, Hu C, et al. Unveiling the power of Haar frequency domain: Advancing small target motion detection in dim light[J]. Applied Soft Computing, 2024, 167: 112281.
    '''

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        'sigma1'    : 'retina.sigma',
        'n1'        : 'lamina.order1',
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order2',
        'tau2'      : 'lamina.tau2',
        'a'         : 'medulla.temporal_kernel_len',
        'r'         : 'medulla.spatial_kernel_size',
        'n3'        : 'medulla.delay_on.order',
        'tau3'      : 'medulla.delay_on.tau',
        'n4'        : 'medulla.delay_off.order',
        'tau4'      : 'medulla.delay_off.tau',
        'sigma2'    : 'lobula.spatial_inhibition.sigma1',
        'sigma3'    : 'lobula.spatial_inhibition.sigma2',
        }
    
    def __init__(self):
        ''' Constructor method '''
        super().__init__()

        self.medulla = haarstmd_core.Medulla()
        self.lobula = haarstmd_core.Lobula()

        # init parameter
        self.retina.sigma = 1
        self.lamina.order1 = 10
        self.lamina.tau1 = 3
        self.lamina.order2 = 10
        self.lamina.tau2 = 9
        self.lobula.spatial_inhibition.sigma1 = 1.5
        self.lobula.spatial_inhibition.sigma2 = 3
    

    def forward(self, x):

        retina_output = self.retina(x)
        lamina_output = self.lamina(retina_output)
        correlated_spatial_output, correlated_temporal_output = self.medulla(lamina_output)
        lobula_output = self.lobula(correlated_spatial_output, 
                                    correlated_temporal_output)

        self.model_output['response'] = compute_response(lobula_output)
        self.model_output['direction'] = compute_direction(lobula_output)

        return self.model_output

        