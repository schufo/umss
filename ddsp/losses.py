# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch implementation of DDSP following closely the original code
# https://github.com/magenta/ddsp/blob/master/ddsp/losses.py

import functools

from ddsp import spectral_ops
from ddsp.core import hz_to_midi
from ddsp.core import safe_divide
from ddsp.core import torch_float32

import numpy as np
import torch


def mean_difference(target, value, loss_type='L1', weights=None):
    """Common loss functions.
    Args:
      target: Target tensor.
      value: Value tensor.
      loss_type: One of 'L1', 'L2', or 'COSINE'.
      weights: A weighting mask for the per-element differences.
    Returns:
      The average loss.
    Raises:
      ValueError: If loss_type is not an allowed value.
    """
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = loss_type.upper()
    if loss_type == 'L1':
        dims = [x for x in range(len(difference.shape))]
        return torch.mean(torch.abs(difference * weights), dim=dims)
    elif loss_type == 'L2':
        dims = [x for x in range(len(difference.shape))]
        return torch.mean(difference**2 * weights, dim=dims)
    # elif loss_type == 'COSINE':
        # return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
    else:
        raise ValueError('Loss type ({}), must be '
                         '"L1", "L2" '.format(loss_type))


class SpectralLoss(torch.nn.Module):
    """Multi-scale spectrogram loss.
    This loss is the bread-and-butter of comparing two audio signals. It offers
    a range of options to compare spectrograms, many of which are redunant, but
    emphasize different aspects of the signal. By far, the most common comparisons
    are magnitudes (mag_weight) and log magnitudes (logmag_weight).
    """

    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 loss_type='L1',
                 mag_weight=1.0,
                 delta_time_weight=0.0,
                 delta_freq_weight=0.0,
                 cumsum_freq_weight=0.0,
                 logmag_weight=0.0,
                 logmel_weight=0.0
                 #loudness_weight=0.0
                 ):
        """Constructor, set loss weights of various components.
        Args:
          fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
            spectrogram has a time-frequency resolution trade-off based on fft size,
            so comparing multiple scales allows multiple resolutions.
          loss_type: One of 'L1', 'L2', (or 'COSINE', not implemented in PyTorch).
          mag_weight: Weight to compare linear magnitudes of spectrograms. Core
            audio similarity loss. More sensitive to peak magnitudes than log
            magnitudes.
          delta_time_weight: Weight to compare the first finite difference of
            spectrograms in time. Emphasizes changes of magnitude in time, such as
            at transients.
          delta_freq_weight: Weight to compare the first finite difference of
            spectrograms in frequency. Emphasizes changes of magnitude in frequency,
            such as at the boundaries of a stack of harmonics.
          cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
            across frequency for each slice in time. Similar to a 1-D Wasserstein
            loss, this hopefully provides a non-vanishing gradient to push two
            non-overlapping sinusoids towards eachother.
          logmag_weight: Weight to compare log magnitudes of spectrograms. Core
            audio similarity loss. More sensitive to quiet magnitudes than linear
            magnitudes.
          loudness_weight: Weight to compare the overall perceptual loudness of two
            signals. Very high-level loss signal that is a subset of mag and
            logmag losses.
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.logmag_weight = logmag_weight
        self.logmel_weight = logmel_weight
        # self.loudness_weight = loudness_weight

        self.spectrogram_ops = []
        for size in self.fft_sizes:
            spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size, add_in_sqrt=1e-10)
            self.spectrogram_ops.append(spectrogram_op)
        if self.logmel_weight > 0:
            self.logmel_op = functools.partial(spectral_ops.compute_logmel, bins=500, add_in_sqrt=1e-10)

    def forward(self, target_audio, audio):

        loss = 0.0

        diff = spectral_ops.diff
        cumsum = torch.cumsum

        # Compute loss for each fft size.
        for loss_op in self.spectrogram_ops:
            target_mag = loss_op(target_audio)
            value_mag = loss_op(audio)

            # Add magnitude loss.
            if self.mag_weight > 0:
                loss += self.mag_weight * mean_difference(target_mag, value_mag,
                                                          self.loss_type)

            if self.delta_time_weight > 0:
                target = diff(target_mag, axis=1)
                value = diff(value_mag, axis=1)
                loss += self.delta_time_weight * mean_difference(
                    target, value, self.loss_type)

            if self.delta_freq_weight > 0:
                target = diff(target_mag, axis=2)
                value = diff(value_mag, axis=2)
                loss += self.delta_freq_weight * mean_difference(
                    target, value, self.loss_type)

            if self.cumsum_freq_weight > 0:
                target = cumsum(target_mag, dim=2)
                value = cumsum(value_mag, dim=2)
                loss += self.cumsum_freq_weight * mean_difference(
                    target, value, self.loss_type)

            # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight > 0:
                target = spectral_ops.safe_log(target_mag)
                value = spectral_ops.safe_log(value_mag)
                loss += self.logmag_weight * mean_difference(target, value,
                                                             self.loss_type)

        if self.logmel_weight > 0:
            target = self.logmel_op(target_audio)[:250, :]
            value = self.logmel_op(audio)[:250, :]
            loss += self.logmel_weight * mean_difference(target, value, self.loss_type)

        return loss


class LSFRegularizer(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, lsf):
        """

        Args:
            lsf: shape [batch_size, n_frames, n_lsf]

        Returns:

        """

        delta_lsf = spectral_ops.diff(lsf, axis=-1)

        delta_lsf_l2 = delta_lsf.norm(p=2, dim=-1)

        dims = [x for x in range(len(delta_lsf_l2.shape))]

        loss = torch.mean(delta_lsf_l2, dim=dims)

        return loss

