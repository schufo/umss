import json
from pathlib import Path
import os

import torch
import numpy as np

import model_utls
import ddsp.spectral_ops, ddsp.core

import librosa
import soundfile
import matplotlib.pyplot as plt

def get_device(use_accelerator=True):
    """Return 'cuda', 'mps' or 'cpu', preferring an available accelerator unless use_accelerator is False."""
    if use_accelerator and torch.cuda.is_available():
        return 'cuda'
    if use_accelerator and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_info(path):
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = sfi.frames
    info['duration'] = sfi.duration
    return info


def load_audio(path, start=0, dur=None):
    """Load audio as a float32 tensor of shape [channels, samples].

    start and dur are given in seconds; dur=None loads until the end of the file.
    """
    samplerate = soundfile.info(path).samplerate
    start = int(start * samplerate)
    stop = start + int(dur * samplerate) if dur else None
    audio, _ = soundfile.read(path, always_2d=True, start=start, stop=stop, dtype='float32')
    return torch.from_numpy(audio.T.copy())


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(
        state, is_best, path, tag
):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, tag + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, tag + '.pth')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


def load_model(tag, device='cpu', return_args=False):
    """

    """
    model_path = 'trained_models/{}'.format(tag)
    # load model from disk
    with open(Path(model_path, tag + '.json'), 'r') as stream:
        results = json.load(stream)

    target_model_path = next(Path(model_path).glob("%s*.pth" % tag))
    state = torch.load(
        target_model_path,
        map_location=device,
        weights_only=True
    )

    architecture = results['args']['architecture']
    model_class = model_utls.ModelLoader.get_model(architecture)
    trained_model = model_class.from_config(results['args'])

    trained_model.load_state_dict(state)
    trained_model.eval()
    trained_model.to(device)

    if return_args: return trained_model, results['args']
    else: return trained_model



def worker_init_fn(worker_id):
    """
    Set numpy seed for each work so that each worker has a different seed,
    the worker seeds are different at each epoch, but the seeds are the same at each
    run with the same torch.manual_seed.

    """
    seed = torch.initial_seed()

    max_numpy_seed = 2**32 - 1
    if seed > max_numpy_seed:
        # use only the 9 least significant digits
        seed = 1e-9 * seed
        x = int(seed)
        seed = int((seed - x) * 1e9)

    # add worker_id to compensate too low precision (avoid that all workers have same numpy seed)
    seed += worker_id

    np.random.seed(seed)


def masking_from_synth_signals_torch(true_mix, estimated_sources, n_fft=2048, n_hop=256):
    """

    Args:
        true_mix: torch.Tensor [batch_size, n_samples]
        estimated_sources: [batch_size, n_sources, n_samples]

    Returns:
        source_estimate_time_domain: torch.Tensor [batch_size * n_sources, n_samples],
            estimated sources obtained by masking
    """

    batch_size, n_sources, n_samples = estimated_sources.shape
    overlap = 1 - n_hop / n_fft
    estimated_sources = estimated_sources.reshape((batch_size * n_sources, -1))

    # compute complex stft of true mix
    mix_stft = ddsp.spectral_ops.stft(true_mix, frame_size=n_fft, overlap=overlap, center=True, pad_end=False)
    mix_stft = mix_stft.repeat_interleave(n_sources, dim=0)
    mix_stft = torch.view_as_complex(mix_stft)

    # magnitude stfts of estimated sources
    sources_mag = ddsp.spectral_ops.compute_mag(estimated_sources, n_fft, overlap, center=True, pad_end=False)

    _, n_freqs, n_frames = sources_mag.shape  # [batch_size * n_sources, n_freqs, n_frames]

    # compute mix estimates as sum of the sources' magnitude spectrograms
    estimated_mix_mag = torch.sum(sources_mag.reshape((batch_size, n_sources, n_freqs, n_frames)), dim=1)
    estimated_mix_mag = estimated_mix_mag.repeat_interleave(n_sources, dim=0)  # [batch_size * n_sources, n_freqs, n_frames]

    mask = sources_mag / (estimated_mix_mag + 1e-12)
    source_stft = mask * mix_stft
    source_stft = torch.view_as_real(source_stft)

    source_estimate_time_domain = ddsp.spectral_ops.istft(source_stft, n_fft, overlap, center=True, length=n_samples)

    return source_estimate_time_domain


def masking_from_synth_signals(true_mix, estimated_sources, n_fft=2048, n_hop=256):
    """

    Args:
        true_mix: torch.Tensor [batch_size, n_samples]
        estimated_sources: [batch_size, n_sources, n_samples]

    Returns:

    """
    true_mix = true_mix.numpy()[0, :]
    estimated_sources = estimated_sources.numpy()[0, :, :]

    n_sources, n_samples = estimated_sources.shape

    # get magnitude and phase specrogram of true mix
    mix_stft = librosa.stft(true_mix, n_fft=n_fft, hop_length=n_hop, pad_mode='reflect')

    source_estimates_mag = []
    for s in range(n_sources):
        source_mag = abs(librosa.stft(estimated_sources[s, :], n_fft=n_fft, hop_length=n_hop, pad_mode='reflect'))
        source_estimates_mag.append(source_mag)

    estimated_mix_mag = sum(source_estimates_mag)

    source_estimates_masking = []
    for s in range(n_sources):
        mask = source_estimates_mag[s] / (estimated_mix_mag + 1e-12)
        source_stft = mask * mix_stft
        source_estimate_time_domain = librosa.istft(source_stft, hop_length=n_hop, win_length=n_fft, length=n_samples)
        source_estimates_masking.append(source_estimate_time_domain)

    return source_estimates_masking


def masking_unets_softmasks(trained_model, mix, f0_info, n_sources, n_fft=1024, n_hop=256):
    """
    Compute the softmasks after obtaining all source estimates and use their sum as mix estimate.
    Args:
        trained_model: trained U-Net with loaded weights
        mix: torch.Tensor, shape [batch_size, n_samples] input mixture
        f0_info: torch.Tensor, shape [batch_size, n_frames, n_sources]
        n_sources: int, number of sources
        n_fft: int, FFT length used in the model (the same is used for STFT and iSTFT in this function)
        n_hop: int, window hop length used in the model (the same is used for STFT and iSTFT in this function)

    Returns:
        estimated_sources, list of source estimates in time domain
    """

    estimated_sources = []
    estimates_mag_sources = []
    for s in range(n_sources):
        f0_target = f0_info[:, :, s].unsqueeze(2)
        with torch.no_grad():
            mag_estimate = trained_model((mix, f0_target))[0, 0, :, :]
            estimates_mag_sources.append(mag_estimate.cpu().numpy())

    mix_estimate = sum(estimates_mag_sources)

    mix_padded = ddsp.core.pad_for_stft(mix, n_fft, n_hop)
    mix_padded = mix_padded.cpu().numpy()[0, :]
    mix_stft = librosa.stft(mix_padded, n_fft=n_fft, hop_length=n_hop, center=True, pad_mode='reflect')

    for s in range(n_sources):
        mask = estimates_mag_sources[s] / (mix_estimate + 1e-12)
        source_stft = mask * mix_stft
        source_estimate_time_domain = librosa.istft(source_stft, hop_length=n_hop, win_length=n_fft, center=True, length=mix.shape[1])
        estimated_sources.append(source_estimate_time_domain)

    return estimated_sources


def masking_unets2(trained_model, mix, f0_info, n_sources, n_fft=1024, n_hop=256):

    estimated_sources = []
    trained_model.return_mask = True
    for s in range(n_sources):
        f0_target = f0_info[:, :, s].unsqueeze(2)
        with torch.no_grad():
            mask = trained_model((mix, f0_target))[0, 0, :, :]

        audio_padded = ddsp.core.pad_for_stft(mix, n_fft, n_hop)
        audio_padded = audio_padded.cpu().numpy()[0, :]
        mix_complex = librosa.stft(audio_padded, n_fft=n_fft, hop_length=n_hop, center=True, pad_mode='reflect')

        source_estimate_complex = mask * mix_complex
        source_estimate_complex = source_estimate_complex.cpu().numpy()
        source_estimate = librosa.istft(source_estimate_complex, hop_length=n_hop,
                                        win_length=n_fft, center=True, length=mix.shape[1])
        estimated_sources.append(source_estimate)
    return estimated_sources