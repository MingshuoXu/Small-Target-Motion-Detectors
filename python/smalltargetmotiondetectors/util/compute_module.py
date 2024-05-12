import numpy as np
import math

def compute_temporal_conv(iptCell, kernel, pointer=None):
    """
    Computes temporal convolution.

    Parameters:
    - iptCell: A list of arrays where each element has the same dimension.
    - kernel: A vector representing the convolution kernel.
    - headPointer: Head pointer of the input cell array (optional).

    Returns:
    - optMatrix: The result of the temporal convolution.
    """

    # Default value for headPointer
    if pointer is None:
        pointer = len(iptCell) - 1

    # Initialize output matrix
    if iptCell[pointer] is None:
        return None

    # Ensure kernel is a vector
    kernel = np.squeeze(kernel)
    if not np.ndim(kernel) == 1:
        raise ValueError('The kernel must be a vector.')

    # Determine the lengths of input cell array and kernel
    k1 = len(iptCell)
    k2 = len(kernel)
    length = min(k1, k2)

    optMatrix = np.zeros_like(iptCell[pointer])
    # Perform temporal convolution
    for t in range(length):
        j = (pointer - t) % k1
        if np.abs(kernel[t]) > 1e-16 and iptCell[j] is not None:
            optMatrix += iptCell[j] * kernel[t]

    return optMatrix


def compute_circularlist_conv(circularCell, temporalKernel):
    """
    Compute the convolution of a circular cell with a temporal kernel.
    
    Args:
    - circularCell: The circular cell data.
    - temporalKernel: The temporal kernel data.
    
    Returns:
    - opt_matrix: The result of the convolution.
    """
    optMatrix = compute_temporal_conv(circularCell, 
                                      temporalKernel, 
                                      circularCell.pointer )
    return optMatrix


def compute_response(ipt):
    """
    Computes the maximum response from multiple inputs.

    Parameters:
    - ipt: List containing input data.

    Returns:
    - response: Maximum response computed from the inputs.
    """
    k = len(ipt)
    response = ipt[0]

    # Compute maximum response
    if k > 1:
        for idx in range(1, k):
            response = np.maximum(response, ipt[idx])

    return response


def compute_direction(ipt):
    """
    Compute the dominant direction given a set of directional responses

    Parameters:
    - ipt: List containing directional responses.

    Returns:
    - directionOpt: Dominant direction computed from the responses.
    """
    # Get the number of directions
    numDirection = len(ipt)

    # Get the size of the input matrix
    m, n = ipt[0].shape

    # Initialize variables for cosine and sine components of the direction response
    outputCos = np.zeros((m, n))
    outputSin = np.zeros((m, n))

    # Compute the weighted sum of cosine and sine components for each direction
    for idx, iptDire in enumerate(ipt):
        outputCos += iptDire * math.cos(idx * 2 * math.pi / numDirection)
        outputSin += iptDire * math.sin(idx * 2 * math.pi / numDirection)

    # Compute the direction based on the arctan2 function
    directionOpt = np.arctan2(outputSin, outputCos)

    # Adjust directions to be in the range [0, 2*pi]
    directionOpt[directionOpt < 0] += 2 * math.pi

    # Set directions where both sine and cosine components are zero to NaN
    nonId = ~(outputSin.astype(bool) & outputCos.astype(bool))
    directionOpt[nonId] = np.nan

    return directionOpt


def slice_matrix_holding_size(iptMatrix, shiftX, shiftY):
    """
    Slice the input matrix while maintaining its size by circular shifting.

    Parameters:
    - input_mat: Input matrix.
    - shiftX: Shift value along the x-axis.
    - shiftY: Shift value along the y-axis.

    Returns:
    - Opt: Sliced matrix holding the original size.
    """
    # Round shift values to integers
    shiftX = round(shiftX)
    shiftY = round(shiftY)

    # Get the size of the input matrix
    m, n = iptMatrix.shape

    # If the shifts exceed the matrix dimensions, return a matrix of zeros
    if abs(shiftX) >= n or abs(shiftY) >= m:
        return np.zeros((m, n))

    # Perform circular shift on the input matrix
    Opt = np.roll(iptMatrix, (shiftX, shiftY), axis=(1, 0))

    # Set the sliced regions to zero based on the shift direction
    if shiftX > 0:
        Opt[:, :shiftX] = 0
    else:
        Opt[:, shiftX:] = 0

    if shiftY > 0:
        Opt[:shiftY, :] = 0
    else:
        Opt[shiftY:, :] = 0

    return Opt








