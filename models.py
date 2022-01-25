import torch

import network_components as nc
from ddsp import spectral_ops, synths, core
from model_utls import _Model

import matplotlib.pyplot as plt

import librosa as lb
from scipy.ndimage import filters
import numpy as np



class SourceFilterMixtureAutoencoder2(_Model):

    """Autoencoder that encodes a mixture of n voices into synthesis parameters
    from which the mixture is re-synthesised. Synthesis of each voice is done with a
    source filter model """

    def __init__(self,
                 n_harmonics=101,
                 filter_order=20,
                 fft_size=512,
                 hop_size=256,
                 n_samples=64000,
                 return_sources=False,
                 harmonic_roll_off=12,
                 estimate_noise_mag=False,
                 f_ref=200,  # for harmonics roll off
                 encoder='SeparationEncoderSimple',
                 encoder_hidden_size=256,
                 embedding_size=128,
                 decoder_hidden_size=512,
                 decoder_output_size=512,
                 n_sources=2,
                 bidirectional=True,
                 voiced_unvoiced_diff=True
                 ):

        super().__init__()

        if harmonic_roll_off == -1:
            # estimate roll off
            output_splits=(('harmonic_amplitude', 1),
                           ('noise_gain', 1),
                           ('line_spectral_frequencies', filter_order + 1),
                           ('harmonic_roll_off', 1))
        else:
            output_splits=(('harmonic_amplitude', 1),
                           ('noise_gain', 1),
                           ('line_spectral_frequencies', filter_order + 1))

        # attributes
        self.return_sources = return_sources
        self.n_harmonics = n_harmonics
        self.output_splits = output_splits
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.return_synth_controls = False
        self.harmonic_roll_off = harmonic_roll_off
        self.f_ref = torch.tensor(f_ref, dtype=torch.float32)
        self.estimate_noise_mag = estimate_noise_mag
        self.return_lsf = False
        self.voiced_unvoiced_diff = voiced_unvoiced_diff

        # neural networks
        overlap = hop_size / fft_size

        if encoder == 'MixEncoderSimple':
            self.encoder = nc.MixEncoderSimple(fft_size=fft_size, overlap=overlap,
                                               hidden_size=encoder_hidden_size, embedding_size=embedding_size,
                                               n_sources=n_sources, bidirectional=bidirectional)

        self.decoder = nc.SynthParameterDecoderSimple(z_size=embedding_size,
                                                      hidden_size=decoder_hidden_size,
                                                      output_size=decoder_output_size,
                                                      bidirectional=bidirectional)
        self.dense_outs = torch.nn.ModuleList([torch.nn.Linear(decoder_output_size, v[1]) for v in output_splits])

        if self.harmonic_roll_off == -2:
            self.gru_roll_off = torch.nn.GRU(decoder_output_size, 1, batch_first=True)
        if self.estimate_noise_mag:
            self.gru_noise_mag = torch.nn.GRU(decoder_output_size, 40, batch_first=True)

        # synth
        self.source_filter_synth = synths.SourceFilterSynth2(n_samples=n_samples,
                                                             sample_rate=16000,
                                                             n_harmonics=n_harmonics,
                                                             audio_frame_size=fft_size,
                                                             hp_cutoff=500,
                                                             f_ref=f_ref,
                                                             estimate_voiced_noise_mag=estimate_noise_mag)


    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        filter_order = config['filter_order'] if 'filter_order' in keys else 10
        harmonic_roll_off = config['harmonic_roll_off'] if 'harmonic_roll_off' in keys else 12
        f_ref = config['f_ref_source_spec'] if 'f_ref_source_spec' in keys else 500
        encoder = config['encoder'] if 'encoder' in keys else 'SeparationEncoderSimple'
        encoder_hidden_size = config['encoder_hidden_size'] if 'encoder_hidden_size' in keys else 256
        embedding_size = config['embedding_size'] if 'embedding_size' in keys else 128
        decoder_hidden_size = config['decoder_hidden_size'] if 'decoder_hidden_size' in keys else 512
        decoder_output_size = config['decoder_output_size'] if 'decoder_output_size' in keys else 512
        n_sources = config['n_sources'] if 'n_sources' in keys else 2
        estimate_noise_mags = config['estimate_noise_mags'] if 'estimate_noise_mags' in keys else False
        bidirectional = not config['unidirectional'] if 'unidirectional' in keys else True
        voiced_unvoiced_diff = not config['voiced_unvoiced_same_noise'] if 'voiced_unvoiced_same_noise' in keys else True

        return cls(filter_order=filter_order,
                   fft_size=config['nfft'],
                   hop_size=config['nhop'],
                   harmonic_roll_off=harmonic_roll_off,
                   estimate_noise_mag=estimate_noise_mags,
                   f_ref=f_ref,
                   encoder=encoder,
                   encoder_hidden_size=encoder_hidden_size,
                   embedding_size=embedding_size,
                   decoder_hidden_size=decoder_hidden_size,
                   decoder_output_size=decoder_output_size,
                   n_sources=n_sources,
                   bidirectional=bidirectional,
                   voiced_unvoiced_diff=voiced_unvoiced_diff)


    def forward(self, audio, f0_hz):
        # audio [batch_size, n_samples]
        # f0_hz [batch_size, n_freq_frames, n_sources]

        z = self.encoder(audio, f0_hz)  # [batch_size, n_frames, n_sources, embedding_size]

        batch_size, n_frames, n_sources, embedding_size = z.shape

        f0_hz = f0_hz.transpose(1, 2)  # [batch_size, n_sources, n_freq_frames]
        f0_hz = torch.reshape(f0_hz, (batch_size*n_sources, -1))  # [batch_size * n_sources, n_freq_frames]

        f0_hz = core.resample(f0_hz, n_frames)

        if self.voiced_unvoiced_diff:
            # use different noise models for voiced and unvoiced frames (this option was not used in the experiments)
            voiced_unvoiced = torch.where(f0_hz > 0., torch.tensor(1., device=f0_hz.device),
                                                  torch.tensor(0., device=f0_hz.device))[:, :, None]
        else:
            # one noise model (this option was used in the experiments for the paper)
            voiced_unvoiced = torch.ones_like(f0_hz)[:, :, None]

        f0_hz = f0_hz[:, :, None]  # [batch_size * n_sources, n_frames, 1]

        z = z.permute(0, 2, 1, 3)
        z = z.reshape((batch_size*n_sources, n_frames, embedding_size))

        x = self.decoder(f0_hz, z)


        outputs = {}
        for layer, (key, _) in zip(self.dense_outs, self.output_splits):
            outputs[key] = layer(x)

        if self.harmonic_roll_off == -1:
            harmonic_roll_off = core.exp_sigmoid(outputs['harmonic_roll_off'], max_value=20.)
        elif self.harmonic_roll_off == -2:
            # constant value for roll off is GRU output of last frame through exponential sigmoid activation
            harmonic_roll_off = core.exp_sigmoid(4 * self.gru_roll_off(x)[0][:, -1, :], max_value=15., exponent=2.)
            harmonic_roll_off = torch.ones_like(outputs['harmonic_amplitude']) * harmonic_roll_off[:, :, None]
        else:
            harmonic_roll_off = torch.ones_like(outputs['harmonic_amplitude']) * self.harmonic_roll_off

        if self.estimate_noise_mag:
            noise_mag = self.gru_noise_mag(x)[0][:, -1, :]
            noise_mag = noise_mag[:, None, :]
        else:
            noise_mag = None

        # return synth controls for insights into how the input sources are reconstructed
        if self.return_synth_controls:
            return self.source_filter_synth.get_controls(outputs['harmonic_amplitude'],
                                                         harmonic_roll_off,
                                                         f0_hz,
                                                         outputs['noise_gain'],
                                                         voiced_unvoiced,
                                                         outputs['line_spectral_frequencies'],
                                                         noise_mag)


        # apply synthesis model
        signal = self.source_filter_synth(outputs['harmonic_amplitude'],
                                          harmonic_roll_off,
                                          f0_hz,
                                          outputs['noise_gain'],
                                          voiced_unvoiced,
                                          outputs['line_spectral_frequencies'],
                                          noise_mag)

        sources = torch.reshape(signal, (batch_size, n_sources, -1))
        mix = torch.sum(sources, dim=1)

        if self.return_sources:
            return mix, sources
        if self.return_lsf:
            lsf = core.lsf_activation(outputs['line_spectral_frequencies'])
            return mix, lsf
        return mix




# -------- U-Net Baselines -------------------------------------------------------------------------------------------

class NormalizeSpec(torch.nn.Module):
    def __init__(self):
        super(NormalizeSpec, self).__init__()

    def forward(self, spectrogram):
        """
        Input: spectrograms
              (nb_samples, nb_channels, nb_bins, nb_frames)
        Returns: normalized spectrograms (divided by their respective max)
              (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        max_values, max_idx = torch.max(spectrogram, dim=2, keepdim=True)
        max_values, max_idx = torch.max(max_values, dim=3, keepdim=True)

        max_values[max_values == 0, ...] = 1

        norm_spec = spectrogram / max_values
        return norm_spec


class ConditionGenerator(torch.nn.Module):

    """
    Process f0 information in a more efficient way
    """
    def __init__(self,
                 n_fft,
                 overlap):
        super().__init__()

        self.n_fft = n_fft
        self.overlap = overlap
        self.harmonic_synth = synths.Harmonic(n_samples=64000)

        in_features = int(n_fft//2 + 1)

        self.linear_gamma_1 = torch.nn.Linear(in_features, in_features)
        self.linear_gamma_2 = torch.nn.Linear(in_features, in_features)

        self.linear_beta_1 = torch.nn.Linear(in_features, in_features)
        self.linear_beta_2 = torch.nn.Linear(in_features, in_features)

    def forward(self, f0_hz):

        # f0_hz with shape [batch_size, n_frames, 1] contains f0 in Hz per frame for target source
        batch_size, n_frames, _ = f0_hz.shape
        device = f0_hz.device
        harmonic_amplitudes = torch.ones((batch_size, n_frames, 1), dtype=torch.float32, device=device)
        harmonic_distribution = torch.ones((batch_size, n_frames, 101), dtype=torch.float32, device=device)
        harmonic_signal = self.harmonic_synth(harmonic_amplitudes, harmonic_distribution, f0_hz)

        harmonic_mag = spectral_ops.compute_mag(harmonic_signal, self.n_fft, self.overlap, pad_end=True, center=True)
        harmonic_mag = harmonic_mag.transpose(1, 2)  # [batch_size, n_frames, n_features]

        gamma = self.linear_gamma_1(harmonic_mag)
        gamma = torch.tanh(gamma)
        gamma = self.linear_gamma_2(gamma)
        gamma = torch.relu(gamma)
        gamma = gamma.transpose(1, 2)  # [batch_size, n_features, n_frames]

        beta = self.linear_beta_1(harmonic_mag)
        beta = torch.tanh(beta)
        beta = self.linear_beta_2(beta)
        beta = torch.relu(beta)
        beta = beta.transpose(1, 2)  # [batch_size, n_features, n_frames]

        return beta, gamma


def process_f0(f0, f_bins, n_freqs):
    freqz = np.zeros((f0.shape[0], f_bins.shape[0]))
    haha = np.digitize(f0, f_bins) - 1
    idx2 = haha < n_freqs
    haha = haha[idx2]
    freqz[range(len(haha)), haha] = 1
    atb = filters.gaussian_filter1d(freqz.T, 1, axis=0, mode='constant').T
    min_target = np.min(atb[range(len(haha)), haha])
    atb = atb / min_target
    atb[atb > 1] = 1
    return atb

def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins


def gaussian_kernel1d(sigma, truncate=4.0):
    """
    Computes a 1-D Gaussian convolution kernel.

    Args:
        sigma: standard deviation
        truncate: Truncate the filter at this many standard deviations.

    Returns:
        phi_x: Gaussian kernel

    """

    radius = int(truncate * sigma + 0.5)

    exponent_range = np.arange(1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x



class ConditionGeneratorOriginal(torch.nn.Module):

    """
    Process the f0 information exactly as done in "Petermann et al.,
    DEEP LEARNING BASED SOURCE SEPARATION APPLIED TO CHOIRENSEMBLES, ISMIR 2020"
    """

    def __init__(self,
                 n_fft,
                 overlap):
        super().__init__()

        self.n_fft = n_fft
        self.overlap = overlap

        self.gaussian_kernel = torch.tensor(gaussian_kernel1d(sigma=1.), dtype=torch.float32)[None, None, :]

        self.conv1 = torch.nn.Conv1d(361, 16, kernel_size=10, stride=1, padding=4)
        self.conv2 = torch.nn.Conv1d(16, 64, kernel_size=10, stride=1, padding=4)
        self.conv3 = torch.nn.Conv1d(64, 256, kernel_size=10, stride=1, padding=4)

        self.linear_gamma = torch.nn.Linear(256, 513)
        self.linear_beta = torch.nn.Linear(256, 513)


    def forward(self, f0_hz):

        # f0_hz with shape [batch_size, n_frames, 1] contains f0 in Hz per frame for target source
        batch_size, n_frames, _ = f0_hz.shape
        device = f0_hz.device

        # compute bin index for each f0 value
        k = torch.round(torch.log2(f0_hz/32.7 + 1e-8) * 60) + 1
        k = torch.where(k < 0, torch.tensor(0., device=device), k)
        k = k.type(torch.long)

        f0_one_hot = torch.zeros((batch_size, n_frames, 361), device=device, dtype=torch.float32)
        ones = torch.ones_like(k, device=device, dtype=torch.float32)
        f0_one_hot.scatter_(dim=2, index=k, src=ones)

        padding = self.gaussian_kernel.shape[-1] // 2
        f0_one_hot = f0_one_hot.reshape((batch_size * n_frames, 361))[:, None, :]

        f0_blured = torch.nn.functional.conv1d(f0_one_hot, self.gaussian_kernel.to(device), padding=padding)
        f0_blured = f0_blured.reshape((batch_size, n_frames, -1))
        f0_blured = f0_blured / f0_blured.max(dim=2, keepdim=True)[0]
        f0_blured = f0_blured.transpose(1, 2)  # [batch_size, n_channels, n_frames]

        f0_blured = torch.nn.functional.pad(f0_blured, pad=(0, 1))
        x = self.conv1(f0_blured)
        x = torch.nn.functional.pad(x, pad=(0, 1))
        x = self.conv2(x)
        x = torch.nn.functional.pad(x, pad=(0, 1))
        x = self.conv3(x)  # [batch_size, 256, n_frames]

        x = x.transpose(1, 2)  # [batch_size, n_frames, 256]

        beta = self.linear_beta(x)
        beta = beta.transpose(1, 2)
        gamma = self.linear_gamma(x)
        gamma = gamma.transpose(1, 2)

        return beta, gamma



class BaselineUnet(_Model):

    def __init__(
            self,
            n_fft=1024,
            n_hop=512,
            nb_channels=1,
            sample_rate=16000,
            power=1,
            original=False
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)

        Output: Power/Mag Spectrogram
                (nb_samples, nb_bins, nb_frames, nb_channels)
        """

        super().__init__()

        self.return_mask = False

        self.normalize = NormalizeSpec()

        self.n_fft = n_fft
        self.overlap = 1 - n_hop / n_fft

        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        #self.transform = torch.nn.Sequential(self.stft, self.spec, self.normalize)

        if original:
            self.condition_generator = ConditionGeneratorOriginal(n_fft=n_fft, overlap=self.overlap)
        else:
            self.condition_generator = ConditionGenerator(n_fft=n_fft, overlap=self.overlap)


        # Define the network components
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(True)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(True)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(True)
        )
        self.deconv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.5)
        )
        self.deconv2 = torch.nn.ConvTranspose2d(512, 128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.5)
        )
        self.deconv3 = torch.nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.5)
        )
        self.deconv4 = torch.nn.ConvTranspose2d(128, 32, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            #torch.nn.Dropout2d(0.5)
        )
        self.deconv5 = torch.nn.ConvTranspose2d(64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = torch.nn.Sequential(
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            #torch.nn.Dropout2d(0.5)
        )
        self.deconv6 = torch.nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(2, 2), padding=2)


    @classmethod
    def from_config(cls, config: dict):
        keys = config.keys()
        original = config['original_cu_net'] if 'original_cu_net' in keys else False
        return cls(
                   n_fft=config['nfft'],
                   n_hop=config['nhop'],
                   sample_rate=config['samplerate'],
                   original=original
                   )

    def forward(self, x):

        mix = x[0]  # mix [batch_size, n_samples]
        f0_info = x[1]  #

        beta, gamma = self.condition_generator(f0_info)

        mix_mag = spectral_ops.compute_mag(mix, self.n_fft, self.overlap, pad_end=True, center=True)[:, None, :, :]

        # input must have shape  (batch_size, nb_channels, nb_bins, nb_frames)
        mix_mag_normalized = self.normalize(mix_mag)

        mix_mag_conditioned = mix_mag_normalized * gamma[:, None, :, :] + beta[:, None, :, :]

        conv1_out = self.conv1(mix_mag_conditioned)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size=conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size=conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size=conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size=conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size=mix_mag.size())
        mask = torch.sigmoid(deconv6_out)

        y_hat = mask * mix_mag

        # at test time, either return the mask and multiply with complex mix STFT or compute magnitude estimates
        # for all sources and build soft masks with them and multiply then with complex mix STFT
        if self.return_mask:
            return mask
        else:
            return y_hat


if __name__ == "__main__":
    torch.random.manual_seed(0)
    # model = SourceFilterMixtureAutoencoder2(harmonic_roll_off=-2, estimate_noise_mag=True, bidirectional=False)
    # audio = torch.rand((16, 64000))
    # f0 = torch.rand((16, 125, 2))
    # out = model(audio, f0)
    # print(out.shape)

    model = BaselineUnet(n_fft=1024, n_hop=256, original=True)
    mix = torch.rand((16, 64000))
    info = torch.rand((16, 254, 1)) * 500
    out = model((mix, info))
    print(out.shape)