import torch
import numpy as np


def window_sumsquare(window_tensor, n_frames, hop_length=160):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window_tensor : tensor of window function calculated values
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    Returns
    -------
    wss : torch.tensor, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """

    n_fft = window_tensor.size()[0]
    n = n_fft + hop_length * (n_frames - 1)
    x = torch.zeros(n, dtype=torch.float)

    # Compute the squared window at the desired length
    win_sq = window_tensor.pow(2)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]

    return x


def onnx_atan2(y, x):
    # Create a pi tensor with the same device and data type as y
    pi = torch.tensor(np.pi, device=y.device, dtype=y.dtype)
    half_pi = pi / 2
    eps = 1e-6

    # Compute the arctangent of y/x
    ans = torch.atan(y / (x + eps))

    # Create boolean tensors representing positive, negative, and zero values of y and x
    y_positive = y >= 0
    y_negative = y < 0
    x_negative = x < 0
    x_zero = x == 0

    # Adjust ans based on the positive, negative, and zero values of y and x
    ans += torch.where(y_positive & x_negative, pi, torch.zeros_like(ans))  # Quadrants I and II
    ans -= torch.where(y_negative & x_negative, pi, torch.zeros_like(ans))  # Quadrants III and IV
    ans = torch.where(y_positive & x_zero, half_pi, ans)  # Positive y-axis
    ans = torch.where(y_negative & x_zero, -half_pi, ans)  # Negative y-axis

    return ans