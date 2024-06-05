import os
import yaml
import cupy as cp
import numpy as np


def pixel_by_pixel_correlation(tensor):
    """
    Computes pixel-by-pixel correlation across frames of a 3D tensor using GPU acceleration.

    Parameters:
    tensor (3D array): Input tensor of shape (num_frames, height, width).

    Returns:
    2D array: Correlation matrix of shape (height, width).
    """
    # Transfer the tensor to the GPU
    tensor_gpu = cp.array(tensor)

    # Calculate the mean of each pixel across all frames
    mean_gpu = cp.mean(tensor_gpu, axis=0)

    # Subtract the mean from each pixel to get the zero-mean tensor
    zero_mean_tensor_gpu = tensor_gpu - mean_gpu

    # Calculate the numerator of the correlation coefficient
    numerator_gpu = cp.sum(zero_mean_tensor_gpu[:-1] * zero_mean_tensor_gpu[1:], axis=0)

    # Calculate the denominator of the correlation coefficient
    variance_gpu = cp.sum(zero_mean_tensor_gpu ** 2, axis=0)
    denominator_gpu = cp.sqrt(variance_gpu[:-1] * variance_gpu[1:])

    # Calculate the correlation coefficient
    correlation_gpu = numerator_gpu / denominator_gpu

    # Transfer the result back to the CPU
    correlation = cp.asnumpy(correlation_gpu)

    return correlation