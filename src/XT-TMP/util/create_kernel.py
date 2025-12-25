import numpy as np
from scipy.special import gamma


def create_gaussian_kernel(size, sigma):
    # Ensure size is a tuple containing two integers
    if isinstance(size, int):
        size = (size, size)
    elif len(size) != 2:
        raise ValueError("size must be a integer or a tuple containing two integers")

    # Calculate the radius of the Gaussian filter
    radius_x = (size[0] - 1) / 2
    radius_y = (size[1] - 1) / 2

    # Create meshgrid for x and y coordinates
    x, y = np.meshgrid(np.arange(-radius_x, radius_x+1), np.arange(-radius_y, radius_y+1))

    # Compute the Gaussian filter
    gaussianFilter = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the Gaussian filter
    gaussianFilter = gaussianFilter / np.sum(gaussianFilter)
    
    # Set values below a threshold to zero
    gaussianFilter[gaussianFilter < 1e-4] = 0
    
    return gaussianFilter.astype(np.float32)


def create_gamma_kernel(order=100, 
                        tau=25, 
                        wide=None):
    """
    Generates a discretized Gamma vector.

    Parameters:
    - order: The order of the Gamma function.
    - tau: The time constant of the Gamma function.
    - wide: The length of the vector.

    Returns:
    - gammaKernel: The generated Gamma vector.
    """
    if wide is None: wide = int(np.ceil(3 * tau))

    # Ensure wide is at least 2
    if wide <= 1: wide = 2

    # Compute the values of the Gamma vector
    timeList = np.arange(wide)
    gammaKernel = (
        (order * timeList / tau)**order *
        np.exp(-order * timeList / tau) /
        (gamma(order) * tau)
    )

    # Normalize the Gamma vector
    gammaKernel /= np.sum(gammaKernel)
    gammaKernel[gammaKernel < 1e-4] = 0
    gammaKernel /= np.sum(gammaKernel)
    
    return gammaKernel.astype(np.float32)


def create_inhi_kernel_W2(kernelSize=15, 
                          sigma1=1.5, 
                          sigma2=3, 
                          e=1., 
                          rho=0, 
                          A=1, 
                          B=3):

    # Ensure kernelSize is odd
    if kernelSize % 2 == 0:
        kernelSize += 1
    
    # Determine the center of the kernel
    cenX = round(kernelSize / 2)
    cenY = round(kernelSize / 2)
    
    # Generate grid
    shiftX, shiftY = np.meshgrid(np.arange(1, kernelSize + 1) - cenX, 
                                 np.arange(kernelSize, 0, -1) - cenY)
        
    # Generate gauss functions 1 and 2
    gauss1 = (1 / (2 * np.pi * sigma1**2)) \
             * np.exp(-(shiftX**2 + shiftY**2) / (2 * sigma1**2))
    gauss2 = (1 / (2 * np.pi * sigma2**2)) \
             * np.exp(-(shiftX**2 + shiftY**2) / (2 * sigma2**2))
    
    # Generate DoG, subtracting two gaussian functions
    dogFilter = gauss1 - e * gauss2 - rho
    
    # max(x,0)
    positiveComponent = np.maximum(dogFilter, 0)
    negativeComponent = np.maximum(-dogFilter, 0)
    
    # Inhibition Kernel
    inhibitionKernelW2 = A * positiveComponent - B * negativeComponent

    return inhibitionKernelW2.astype(np.float32)


def create_direction_inhi_kernel(KernelSize=8, Sigma1=1.5, Sigma2=3.0):
    """
    Function Description:
    This function generates lateral inhibition kernels along the Theta direction.
    We adopt a one-dimensional DoG as the lateral inhibition kernel function here.
    """

    # Sampling for DoG
    Zero_Point_DoG_X1 = -np.sqrt((np.log(Sigma2/Sigma1)*2*Sigma1**2*Sigma2**2)/(Sigma2**2-Sigma1**2))
    Zero_Point_DoG_X2 = -Zero_Point_DoG_X1
    Min_Point_DoG_X1 = -np.sqrt((3*np.log(Sigma2/Sigma1)*2*Sigma1**2*Sigma2**2)/(Sigma2**2-Sigma1**2))
    Min_Point_DoG_X2 = -Min_Point_DoG_X1

    if KernelSize % 2 == 0:
        KernelSize += 1

    Half_Kernel_Size = (KernelSize-1)//2
    Quarter_Kernel_Size = (KernelSize-1)//4

    Center_Range_DoG = Zero_Point_DoG_X2 - Zero_Point_DoG_X1
    Center_Step = Center_Range_DoG / Half_Kernel_Size

    Surround_Range_DoG = Min_Point_DoG_X2 - Zero_Point_DoG_X2
    Surround_Step = 2 * Surround_Range_DoG / Quarter_Kernel_Size

    X_Smaller = Zero_Point_DoG_X1 - np.arange(Quarter_Kernel_Size, 0, -1) * Surround_Step
    X_Larger = Zero_Point_DoG_X2 + np.arange(1, Quarter_Kernel_Size+1, 1) * Surround_Step
    X_Center = Zero_Point_DoG_X1 + np.arange(0, Half_Kernel_Size + 1) * Center_Step
    X = np.concatenate((X_Smaller, X_Center, X_Larger))

    Gauss1 = (1 / (np.sqrt(2 * np.pi) * Sigma1)) * np.exp(-(X**2) / (2 * Sigma1**2))
    Gauss2 = (1 / (np.sqrt(2 * np.pi) * Sigma2)) * np.exp(-(X**2) / (2 * Sigma2**2))

    Inhibition_Kernel = Gauss1 - Gauss2

    Inhibition_Kernel[np.abs(Inhibition_Kernel) < 1e-4] = 0

    directionalInhiKernel = np.reshape(Inhibition_Kernel, (1, 1, KernelSize))

    return directionalInhiKernel


def create_T1_kernels(filterNum=4, 
                      alpha=3.0, 
                      eta=1.5, 
                      filterSize=11):

    # If the filter size is even, force it to be odd
    if filterSize % 2 == 0:
        filterSize += 1  

    # Compute angles for each filter
    Theta = np.zeros(filterNum)
    for i in range(filterNum):
        Theta[i] = (i) * np.pi / filterNum

    # Initialize list to store generated kernels
    dictKernel = []

    # Generate coordinates
    r = filterSize // 2
    X, Y = np.meshgrid(np.arange(-r, r + 1), np.arange(r, -r - 1, -1))

    # Generate kernels for each filter
    for idx in range(filterNum):
        # Determine the centers of the two Gaussian functions
        X1 = X - alpha * np.cos(Theta[idx])
        Y1 = Y - alpha * np.sin(Theta[idx])
        X2 = X + alpha * np.cos(Theta[idx])
        Y2 = Y + alpha * np.sin(Theta[idx])

        # Generate the two Gaussian functions
        gauss1 = (1 / (2 * np.pi * eta**2)) * np.exp(-(X1**2 + Y1**2) / (2 * eta**2))
        gauss2 = (1 / (2 * np.pi * eta**2)) * np.exp(-(X2**2 + Y2**2) / (2 * eta**2))

        # Create the filter as the difference between the two Gaussians
        dictKernel.append(gauss1 - gauss2)

    return dictKernel


def create_fracdiff_kernel(alpha=0.8, wide=3):
    """
    Generates a fractional difference kernel.

    Parameters:
    - alpha: The fractional difference parameter.
    - wide: The width of the kernel.

    Returns:
    - frackernel: The fractional difference kernel.
    """
    # Ensure the width is at least 2
    if wide < 2:
        wide = 2
    
    # Initialize the kernel
    frackernel = np.zeros(wide)
    
    # Generate the kernel based on alpha
    if alpha == 1:
        frackernel[0] = 1
    elif 0 < alpha < 1:
        t_list = np.arange(wide)
        frackernel = np.exp(-alpha * t_list / (1 - alpha)) / (1 - alpha)

        # Normalize the kernel
        sum_kernel = np.sum(frackernel)  # 1/M(\alpha)
        frackernel = frackernel / sum_kernel
        frackernel[frackernel < 1e-16] = 0
    else:
        raise ValueError("Alpha must be in the interval (0,1].")

    return frackernel.astype(np.float32)


def create_attention_kernel(kernel_size=17, 
                            zeta=[2, 2.5, 3, 3.5], 
                            theta=[0, np.pi/4, np.pi/2, np.pi*3/4]
                            ):
    """
    Creates attention kernels.

    Parameters:
    - kernel_size: Size of the kernel.
    - zeta: List of zeta values.
    - theta: List of theta values.

    Returns:
    - attention_kernel: 2D list containing attention kernels.
    """

    # Adjust kernel size if even
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Initialize the 2D list to store the attention kernels
    r = len(zeta)
    s = len(theta)
    attention_kernel = [[None] * s for _ in range(r)]

    # Calculate the center of the kernel
    center = (kernel_size - 1) / 2

    # Create meshgrid for kernel indices
    shift_x, shift_y = np.meshgrid(np.arange(kernel_size)        - center, 
                                   np.arange(kernel_size, 0, -1) - center)

    # Generate attention kernels for each combination of Zeta and Theta
    for i in range(r):
        for j in range(s):
            # Formula to generate the attention kernel
            attention_kernel_with_ij = 2 / np.pi / zeta[i]**4 * (
                zeta[i]**2 - (shift_x * np.cos(theta[j] + np.pi/2) 
                              + shift_y * np.sin(theta[j] + np.pi/2))**2
                ) * np.exp(-(shift_x**2 + shift_y**2) / 2 / zeta[i]**2)

            attention_kernel_with_ij[np.abs(attention_kernel_with_ij) < 1e-4] = 0
            # flip the kernel used for np.fliter2D
            attention_kernel_with_ij = np.flip(attention_kernel_with_ij, axis=0)
            attention_kernel_with_ij = np.flip(attention_kernel_with_ij, axis=1)
            attention_kernel[i][j] = attention_kernel_with_ij

    return attention_kernel


def create_prediction_kernel(Vel=0.25, 
                             Delta_t=25, 
                             filter_size=25, 
                             FilterNum=8, 
                             zeta=2, 
                             eta=2.5):
    """
    Creates prediction kernels for motion detection.

    Parameters:
    - Vel: Velocity of the moving object.
    - Delta_t: Time interval.
    - filter_size: Size of the filter.
    - FilterNum: Number of filters.
    - zeta: Zeta parameter.
    - eta: Eta parameter.

    Returns:
    - PredictionKernal: List containing prediction kernels.
    """

    # Initialize the list for prediction kernels
    PredictionKernal = []

    # Create meshgrid
    Center = (filter_size - 1) / 2
    ShiftX, ShiftY = np.meshgrid(np.arange(filter_size) - Center, 
                                 np.arange(filter_size, 0, -1) - Center)

    # Compute angle
    fai = np.arctan2(ShiftY, ShiftX)

    # Compute delta X and delta Y
    Delta_X = Vel * Delta_t * np.cos(fai)
    Delta_Y = Vel * Delta_t * np.sin(fai)

    # Generate prediction kernels
    for idx in range(FilterNum):
        theta = (idx - 1) * 2 * np.pi / FilterNum

        PredictionKernalWithIdx = np.exp(
            -((ShiftX - Delta_X) ** 2 + (ShiftY - Delta_Y) ** 2) / (2 * zeta ** 2)
            + eta * np.cos(fai - theta)
        )

        # normalization
        PredictionKernalWithIdx = PredictionKernalWithIdx / np.sum(PredictionKernalWithIdx)

        # To speed up the calculation
        PredictionKernalWithIdx[PredictionKernalWithIdx < 5e-4] = 0
        PredictionKernalWithIdx /= np.sum(PredictionKernalWithIdx)

        # flip the kernel used for np.fliter2D
        PredictionKernalWithIdx = np.flip(PredictionKernalWithIdx, axis=0)
        PredictionKernalWithIdx = np.flip(PredictionKernalWithIdx, axis=1)
    
        # normalizing again
        PredictionKernal.append(PredictionKernalWithIdx)

    return PredictionKernal

