import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import window_sumsquare, onnx_atan2

class STFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=None,
                 window='hann', center=False):
        """
        This module implements an STFT using 1D convolution and 1D transpose convolutions.
        This is a bit tricky so there are some cases that probably won't work as working
        out the same sizes before and after in all overlap add setups is tough. Right now,
        this code should work with hop lengths that are half the filter length (50% overlap
        between frames).

        Keyword Arguments:
            filter_length {int} -- Length of filters used (default: {512})
            hop_length {int} -- Hop length of STFT (restrict to 50% overlap between frames) (default: {256})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
            window {str} -- Type of window to use (options are only hann now)
                (default: {'hann'})
        """
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.pad_amount = int(self.n_fft / 2)
        self.center = center
        if self.center:
            print(
                "'Center' is not suitable for STFT of long audio, because of shifts on end of res, use it only in stream mode")

        scale = self.n_fft / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.n_fft))

        cutoff = int((self.n_fft / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        # TODO: make different window choice
        if window == "hann":
            self.fft_window = torch.hann_window(n_fft)
        elif window == "bartlett":
            self.fft_window = torch.bartlett_window(n_fft)
        else:
            raise NotImplementedError(f"Window {window} not implemented")

        # window the bases
        forward_basis *= self.fft_window
        inverse_basis *= self.fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        """Take input data (audio) to STFT domain.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (batch_size, num_samples)

        Returns:
            This function returns a complex-valued tensor D with shape (batch_size, nfft // 2 + 1, time, 2) such that

            torch.sqrt(D[..., 0]**2 + D[..., 1]**2) is the magnitudes of frequency bins
            torch.atan2(imag_part.data, real_part.data) is the phases of frequency bins
        """

        # Reflect padding implementation via torch op (instead of F.pad(..., mode="reflect"),
        # because F.pad can't be converted to OpenVino 2022.1. This version is faster as well
        if not self.center:
            input_data = torch.cat([torch.flip(input_data[:, 1:self.pad_amount + 1], [0, 1]), input_data,
                                           torch.flip(input_data[:, -1 * self.pad_amount - 1:-1], [0, 1])], dim=1)
        input_data = input_data.unsqueeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.n_fft / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        concat_tensor = torch.stack((real_part, imag_part), dim=3)

        return concat_tensor

    def inverse(self, input_data):
        """Call the inverse STFT (iSTFT), given complex-valued tensor D with shape (batch_size, nfft // 2 + 1, time, 2)
        by the ```transform``` function.

        Arguments:
            input_data {tensor} -- Fourier image coefficients with shape (batch_size,
                nfft // 2 + 1, time, 2)

        Returns:
            inverse_transform {tensor} -- Reconstructed audio (data). Of
                shape (batch_size, num_samples)
        """

        recombine_real_image_parts = torch.cat((input_data[:, :, :, 0], input_data[:, :, :, 1]), dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_real_image_parts,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        window_sum = window_sumsquare(
            self.fft_window, input_data.size(-2), hop_length=self.hop_length).to(inverse_transform.device)
        # remove modulation effects
        approx_nonzero_indices = (window_sum > torch.finfo(torch.float).tiny).nonzero()
        inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

        # scale by hop ratio
        inverse_transform *= float(self.n_fft) / self.hop_length

        if not self.center:
            inverse_transform = inverse_transform[..., self.pad_amount:-self.pad_amount]
        inverse_transform = inverse_transform.squeeze(1)

        return inverse_transform

    def forward(self, input_data):
        """Take input data (audio) to STFT domain and then back to audio.

        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (batch_size, num_samples)

        Returns:
            reconstruction {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (batch_size, num_samples)
        """
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)

        return reconstruction