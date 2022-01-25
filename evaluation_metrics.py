
import torch
import torchaudio
import numpy as np

import ddsp.spectral_ops


def spectral_snr(target, estimate, eval_frame_length=16000, fft_size=512, overlap=0.5):

    """
    Args:
        target: torch.Tensor of time domain signal(s), shape [batch_size, n_samples]
        estimate: torch.Tensor of time domain signal(s), shape [batch_size, n_samples]

    Returns:
        spectral SNR: torch.Tensor of shape [batch_size, n_eval_frames] containing
            the spectral SNR in dB for each frame.

    """

    assert target.shape == estimate.shape, 'target and estimate must have same shape'

    batch_size, n_samples = target.shape
    n_eval_frames = int(np.ceil(n_samples / eval_frame_length))
    snr_db_frames = torch.zeros((batch_size, n_eval_frames))

    for n in range(n_eval_frames):

        # [batch_size, n_frequencies, n_frames]
        target_mag_spec = ddsp.spectral_ops.compute_mag(target[:, n*eval_frame_length: (n+1)*eval_frame_length],
                                                        size=fft_size, overlap=overlap, center=True)
        estimate_mag_spec = ddsp.spectral_ops.compute_mag(estimate[:, n*eval_frame_length: (n+1)*eval_frame_length],
                                                          size=fft_size, overlap=overlap, center=True)

        snr = torch.sum(target_mag_spec**2, dim=[1,2]) / torch.sum((estimate_mag_spec - target_mag_spec)**2, dim=[1,2])

        snr_db = 10 * torch.log10(snr + 1e-08)

        snr_db_frames[:, n] = snr_db

    return snr_db_frames


def spectral_si_snr(target, estimate, eval_frame_length=16000, fft_size=512, overlap=0.5):

    """
    Args:
        target: torch.Tensor of time domain signal(s), shape [batch_size, n_samples]
        estimate: torch.Tensor of time domain signal(s), shape [batch_size, n_samples]

    Returns:
        spectral SI_SNR: torch.Tensor of shape [batch_size, n_eval_frames] containing
            the spectral scale-invariant SNR in dB for each frame.

    """

    assert target.shape == estimate.shape, 'target and estimate must have same shape'

    batch_size, n_samples = target.shape
    n_eval_frames = int(np.ceil(n_samples / eval_frame_length))
    snr_db_frames = torch.zeros((batch_size, n_eval_frames))

    for n in range(n_eval_frames):

        # [batch_size, n_frequencies, n_frames]
        target_mag_spec = ddsp.spectral_ops.compute_mag(target[:, n*eval_frame_length: (n+1)*eval_frame_length],
                                                        size=fft_size, overlap=overlap, center=True)
        estimate_mag_spec = ddsp.spectral_ops.compute_mag(estimate[:, n*eval_frame_length: (n+1)*eval_frame_length],
                                                          size=fft_size, overlap=overlap, center=True)

        scaler = torch.sum(target_mag_spec * estimate_mag_spec, dim=[1, 2], keepdim=True) / (torch.sum(target_mag_spec**2, dim=[1, 2], keepdim=True) + 1e-12)
        target_mag_spec = target_mag_spec * scaler

        snr = torch.sum(target_mag_spec**2, dim=[1,2]) / torch.sum((estimate_mag_spec - target_mag_spec)**2, dim=[1,2])

        snr_db = 10 * torch.log10(snr + 1e-08)

        snr_db_frames[:, n] = snr_db

    return snr_db_frames


def mel_cepstral_distance(target, estimate, eval_frame_length=16000, fft_size=512, overlap=0.5, device='cpu'):

    assert target.shape == estimate.shape, 'target and estimate must have same shape'

    batch_size, n_samples = target.shape

    n_eval_frames = int(np.ceil(n_samples / eval_frame_length))

    hop_length = int(fft_size * overlap)
    compute_mfccs = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13, log_mels=True,
                                               melkwargs={'n_fft': fft_size, 'hop_length': hop_length}).to(device)

    mcd_frames = torch.zeros((batch_size, n_eval_frames))

    for n in range(n_eval_frames):

        target_mfccs = compute_mfccs(target[:, n*eval_frame_length: (n+1)*eval_frame_length])
        estimate_mfccs = compute_mfccs(estimate[:, n*eval_frame_length: (n+1)*eval_frame_length])

        # root mean square Mel cepstral distance for each FFT frame
        mcd = torch.sqrt(torch.sum((target_mfccs - estimate_mfccs)**2, dim=1))  # [batch_size, n_frames]

        # Mel cepstral distance for the whole evaluation frame (mean of all FFT frame values)
        mcd = torch.mean(mcd, dim=-1)

        mcd_frames[:, n] = mcd

    return mcd_frames


def si_sdr(targets, estimates, eval_frame_length=16000):
    """

     Args:
        targets: torch.Tensor of time domain signal(s), shape [batch_size, n_samples]
        estimates: torch.Tensor of time domain signal(s), shape [batch_size, n_samples]

    Returns:

    """
    assert targets.shape == estimates.shape, 'target and estimate must have same shape'

    batch_size, n_samples = targets.shape
    n_eval_frames = int(np.ceil(n_samples / eval_frame_length))
    si_sdr_db_frames = torch.zeros((batch_size, n_eval_frames))

    for n in range(n_eval_frames):

        eval_frame_targets = targets[:, n*eval_frame_length: (n+1)*eval_frame_length]
        eval_frame_estimates = estimates[:, n*eval_frame_length: (n+1)*eval_frame_length]

        scaler = torch.sum(eval_frame_targets * eval_frame_estimates, dim=1, keepdim=True) / \
                    torch.sum(eval_frame_targets ** 2, dim=1, keepdim=True)

        si_sdr_frame = torch.sum((scaler * eval_frame_targets) ** 2, dim=1) / torch.sum((scaler * eval_frame_targets - eval_frame_estimates) ** 2, dim=1)

        si_sdr_db_frame = 10 * torch.log10(si_sdr_frame + 1e-8)
        si_sdr_db_frames[:, n] = si_sdr_db_frame

    return si_sdr_db_frames
