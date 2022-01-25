# Neural network components

# elements commented with '# DDSP' are re-implemented in PyTorch from
# https://github.com/magenta/ddsp/blob/master/ddsp/training/nn.py

import torch
import numpy as np
from ddsp import core, spectral_ops, synths

import matplotlib.pyplot as plt

# -----------------Normalization -----------------------------------------------
# DDSP
class CustomLayerNorm(torch.nn.Module):
    """Layer normalization
    with different properties of the learnable scale and shift parameters
    than in the original paper and the PyTorch implementation.
    The mean and variance are computed over channels, height, and width for each
    example in a batch. The learned scaling and shifting is the same for all values
    in each channel, instead of being different for each value [ch, h, w].
    This has the advantage that the exact input shape does not need to
    be known when building the model. """

    def __init__(self, ch: int):
        """Args:
            ch: (int) number of channels in input tensor [batch_size, ch, h, w]
        """
        super().__init__()
        self.scale = torch.nn.parameter.Parameter(torch.ones((1, ch, 1, 1)))
        self.shift = torch.nn.parameter.Parameter(torch.zeros((1, ch, 1, 1)))

    def forward(self, x, eps=1e-5):
        """
        Args:
            x: (torch.Tensor) [batch_size, ch, h, w]
        Returns:
            torch.Tensor [batch_size, ch, h, w]
        """
        b, ch, h, w = x.shape
        x = x.reshape((b, ch * h * w))
        var, mean = torch.var_mean(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + eps)
        x = x.reshape((b, ch, h, w))

        return (x * self.scale) + self.shift

# ---------------- Padding -----------------------------------------------------

class SamePadding(torch.nn.Module):
    """Same padding as done in Tensorflow according to
    https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
    In PyTorch, the padding option in conv. layers allows only for padding both sides of a dimension
    with the same number of values.
    """
    def __init__(self, k, s):
        """
        Args:
            k: (int or (int, int)) kernel size
            s: (int or (int, int)) strides
        """
        super().__init__()
        if isinstance(k, int):
            self.k_h = k
            self.k_w = k
        else:
            self.k_h = k[0]
            self.k_w = k[1]
        if isinstance(s, int):
            self.s_h = s
            self.s_w = s
        else:
            self.s_h = s[0]
            self.s_w = s[1]

    def forward(self, x):
        """
        Args:
            x: torch.Tensor [batch_size, n_channels, height, width]
        Returns:
            y: torch.Tensor [batch_size, n_channels, height+pad_h, width+pad_w]
        """
        b, ch, h, w = x.shape

        # compute padding along height
        if h % self.s_h == 0:
            pad_h = max(self.k_h - self.s_h, 0)
        else:
            pad_h = max(self.k_h -(h % self.s_h), 0)

        pad_h_top = int(np.floor(pad_h / 2))
        pad_h_bottom = int(pad_h - pad_h_top)

        # compute padding along width
        if w % self.s_w == 0:
            pad_w = max(self.k_w - self.s_w, 0)
        else:
            pad_w = max(self.k_w -(w % self.s_w), 0)

        pad_w_left = int(np.floor(pad_w / 2))
        pad_w_right = int(pad_w - pad_w_left)

        y = torch.nn.functional.pad(x, pad=(pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))
        return y


# ------------------ ResNet ----------------------------------------------------
# DDSP
def norm_relu_conv(ch_in, ch_out, k, s):
    """Downsample frequency by stride.
    input has shape [batch_size, ch_in, h, w]
    Args:
        ch_in: number of input channels
        ch_out: number of output channels
        w: input width
        h: input height
        k: kernel size
        s: stride over w dimension
    """
    layers = torch.nn.Sequential(
        CustomLayerNorm(ch_in),
        torch.nn.ReLU(),
        SamePadding((k, k), (1, s)),
        torch.nn.Conv2d(ch_in, ch_out, (k, k), stride=(1, s))
    )
    return layers

# DDSP
class ResidualBlock(torch.nn.Module):
    """A Block for ResNet, with a bottleneck.
    x --> Norm, ReLU, (1x1)-conv (ch_in -> ch_b)
      --> Norm, ReLU, (3x3)-conv (ch_b -> ch_b)
      --> Norm, ReLU, (1x1)-conv (ch_b -> 4*ch_b)
      --> out + x
    """

    def __init__(self, ch_in, ch_b, stride, shortcut: bool = False):
        """Downsample frequency by stride
        input to forward() has shape [batch_size, ch_in, h, w]
        Args:
            ch_in: # channels of input
            ch_b: # channels in bottleneck
            stride: stride over w dimension
            shortcut: if True, a shortcut connection AFTER first Norm and ReLU
                      with a Conv2d layer is built to upsample the channel dimenion,
                      otherwise, there is a skip connection from input to output.
        """
        super().__init__()
        ch_out = 4 * ch_b
        self.shortcut = shortcut

        # Layers.
        self.norm_input = CustomLayerNorm(ch_in)
        if self.shortcut:
            self.conv_proj = torch.nn.Sequential(
                SamePadding(k=(1,1), s=(1, stride)),
                torch.nn.Conv2d(ch_in, ch_out, (1, 1), (1, stride))
            )
        self.bottleneck = torch.nn.Sequential(
            SamePadding(k=(1, 1), s=(1, 1)),
            torch.nn.Conv2d(ch_in, ch_b, (1, 1), (1, 1)),
            norm_relu_conv(ch_b, ch_b, k=3, s=stride),
            norm_relu_conv(ch_b, ch_out, k=1, s=1)
        )

    def forward(self, x):
        """x: torch.Tensor [batch_size, ch_in, h, w]"""
        r = x
        x = torch.relu(self.norm_input(x))
        # The projection shortcut should come after the first norm and ReLU
        # since it performs a 1x1 convolution.
        r = self.conv_proj(x) if self.shortcut else r
        x = self.bottleneck(x)
        return x + r

# DDSP
def residual_stack(filters_in,
                   filters_b,
                   stack_sizes,
                   strides):
    """ResNet layers.
    If multiple values are given in the argument lists, multiple residual stacks
    will be concatenated.
    Args:
        filters_in: (list) number of channels of the input
        filters_b: (list) number of filters to apply in the bottleneck
        stack_sizes: (list) number of residual blocks in the stack
        strides: (list) stride over last dimension in the first layer of the stack
    """
    layers = []
    for (ch_in, ch_b, n_layers, stride) in zip(filters_in, filters_b, stack_sizes, strides):
        # Only the first block per residual_stack uses shortcut and strides.
        layers.append(ResidualBlock(ch_in, ch_b, stride, shortcut=True))
        # Add the additional (n_layers - 1) layers to the stack.
        for _ in range(1, n_layers):
            layers.append(ResidualBlock(ch_b*4, ch_b, stride=1, shortcut=False))
    layers.append(CustomLayerNorm(filters_b[-1]*4))
    layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


# DDSP
def resnet(size='small'):
    """Residual network."""
    size_dict = {
        'small': (32, [2, 3, 4]),
        'medium': (32, [3, 4, 6]),
        'large': (64, [3, 4, 6]),
    }
    ch_b, block_size = size_dict[size]
    layers = [
        SamePadding(k=7, s=(1,2)),
        torch.nn.Conv2d(1, 64, (7, 7), (1, 2)),
        SamePadding(k=(1, 3), s=(1, 2)),
        torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
        residual_stack([64, 4*ch_b, 4*2*ch_b], [ch_b, 2*ch_b, 4*ch_b], block_size, [1, 2, 2]),
        residual_stack([4*4*ch_b], [8*ch_b], [3], [2])
    ]
    return torch.nn.Sequential(*layers)


# -------------------- Sinusoidal to Harmonic Encoder ----------------------------------------

# DDSP
def fc(ch_in=256, ch_out=256):
    layers = [
        torch.nn.Linear(ch_in, ch_out),
        torch.nn.LayerNorm(ch_out),  # normalization is done over the last dimension
        torch.nn.LeakyReLU()
    ]
    return torch.nn.Sequential(*layers)

# DDSP
def fc_stack(ch_in=256, ch_out=256, layers=2):
    return torch.nn.Sequential(*([fc(ch_in, ch_out)] + [fc(ch_out, ch_out) for i in range(layers-1)]))

# DDSP
class SinusoidalToHarmonicEncoder(torch.nn.Module):
    """Predicts harmonic controls from sinusoidal controls.
    """

    def __init__(self,
                 fc_stack_layers=2,
                 fc_stack_ch=256,
                 rnn_ch=512,
                 n_sinusoids_in=100,
                 n_harmonics_out=100,
                 amp_scale_fn=core.exp_sigmoid,
                 f0_depth=64,
                 hz_min=20.0,
                 hz_max=1200.0,
                 sample_rate=16000):

        super().__init__()
        self.n_harmonics = n_harmonics_out
        self.amp_scale_fn = amp_scale_fn
        self.f0_depth = f0_depth
        self.hz_min = hz_min
        self.hz_max = hz_max
        self.sample_rate = sample_rate

        # Layers.
        self.pre_rnn = fc_stack(ch_in=n_sinusoids_in*2, ch_out=fc_stack_ch, layers=fc_stack_layers)
        self.rnn = torch.nn.GRU(input_size=fc_stack_ch, hidden_size=rnn_ch, batch_first=True)
        self.post_rnn = fc_stack(ch_in=rnn_ch, ch_out=fc_stack_ch, layers=fc_stack_layers)

        self.amp_out = torch.nn.Linear(fc_stack_ch, 1)
        self.hd_out = torch.nn.Linear(fc_stack_ch, n_harmonics_out)
        self.f0_out = torch.nn.Linear(fc_stack_ch, f0_depth)

    def forward(self, sin_freqs, sin_amps):
        """Converts (sin_freqs, sin_amps) to (f0, amp, hd).
        Args:
          sin_freqs: Sinusoidal frequencies in Hertz, of shape
            [batch, time, n_sinusoids].
          sin_amps: Sinusoidal amplitudes, linear scale, greater than 0, of shape
            [batch, time, n_sinusoids].
        Returns:
          f0: Fundamental frequency in Hertz, of shape [batch, time, 1].
          amp: Amplitude, linear scale, greater than 0, of shape [batch, time, 1].
          hd: Harmonic distribution, linear scale, greater than 0, of shape
            [batch, time, n_harmonics].
        """
        # Scale the inputs.
        nyquist = self.sample_rate / 2.0
        sin_freqs_unit = core.hz_to_unit(sin_freqs, hz_min=0.0, hz_max=nyquist)

        # Combine.
        x = torch.cat([sin_freqs_unit, sin_amps], dim=-1)

        # Run it through the network.
        x = self.pre_rnn(x)
        x = self.rnn(x)[0]  # ignore state output
        x = self.post_rnn(x)

        harm_amp = self.amp_out(x)
        harm_dist = self.hd_out(x)
        f0 = self.f0_out(x)

        # Output scaling.
        harm_amp = self.amp_scale_fn(harm_amp)
        harm_dist = self.amp_scale_fn(harm_dist)
        f0_hz = core.frequencies_softmax(
            f0, depth=self.f0_depth, hz_min=self.hz_min, hz_max=self.hz_max)

        # Filter harmonic distribution for nyquist.
        harm_freqs = core.get_harmonic_frequencies(f0_hz, self.n_harmonics)
        harm_dist = core.remove_above_nyquist(harm_freqs, harm_dist, self.sample_rate)
        harm_dist = core.safe_divide(harm_dist, torch.sum(harm_dist, dim=-1, keepdims=True))

        return (harm_amp, harm_dist, f0_hz)


# -------------------- Encoders ----------------------------------------
class AudioEncoderSimple(torch.nn.Module):

    """Encoder to transform some audio representation into latent representation z"""

    def __init__(self,
                 audio_transform=spectral_ops.compute_logmag,
                 fft_size=1024,
                 overlap=0.5):

        super().__init__()
        self.audio_transform = audio_transform
        self.fft_size = fft_size
        self.overlap = overlap

        self.layer_norm = CustomLayerNorm(fft_size//2 + 1)
        self.gru = torch.nn.GRU(input_size=fft_size//2 + 1, hidden_size=256, batch_first=True)
        self.dense = torch.nn.Linear(256, 128)

    def forward(self, x):
        """

        Args:
            x: audio signal torch.Tensor [batch_size, n_samples]
        Returns:
            z: latent representation torch.Tensor [batch_size, n_frames, n_features]

        """
        # [batch_size, n_frequencies, n_frames]
        x = self.audio_transform(x, fft_size=self.fft_size, overlap=self.overlap)

        # layer norm expects 4-d input [batch_size, n_channel, h, w]
        # the frequencies are the channels and each frequency bin has a dedicated scale and shift parameter
        x = x[:, :, :, None]
        x = self.layer_norm(x)
        x = x.squeeze(-1).transpose(1, 2)  # [batch_size, n_frames, n_features]
        x = self.gru(x)[0]
        z = self.dense(x)
        return z



class MixEncoderSimple(torch.nn.Module):

    """Encoder to transform an audio mixture into a latent representation
     which is then copied in order to obtain as many copies as there are sources
     to separate"""

    def __init__(self,
                 audio_transform=spectral_ops.compute_logmag,
                 fft_size=512,
                 overlap=0.5,
                 hidden_size=256,
                 embedding_size=128,
                 n_sources=2,
                 bidirectional=True):

        super().__init__()
        self.audio_transform = audio_transform
        self.fft_size = fft_size
        self.overlap = overlap
        self.embedding_size = embedding_size
        self.n_sources = n_sources

        if bidirectional: input_size = hidden_size * 2
        else: input_size = hidden_size

        self.layer_norm = CustomLayerNorm(fft_size//2 + 1)
        self.gru1 = torch.nn.GRU(input_size=fft_size//2 + 1, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.gru2 = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.gru3 = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.dense = torch.nn.Linear(input_size, embedding_size)

    def forward(self, x, _):
        """

        Args:
            x: audio signal torch.Tensor [batch_size, n_samples]
            _: dummy argument to allow flexibility when using other encoders
        Returns:
            z1: latent representation of source 1 torch.Tensor [batch_size, n_frames, n_features]
            z2: latent representation of source 2 torch.Tensor [batch_size, n_frames, n_features]

        """
        # [batch_size, n_frequencies, n_frames]
        x = self.audio_transform(x, fft_size=self.fft_size, overlap=self.overlap)

        # layer norm expects 4-d input [batch_size, n_channel, h, w]
        # the frequencies are the channels and each frequency bin has a dedicated scale and shift parameter
        x = x[:, :, :, None]
        x = self.layer_norm(x)
        x = x.squeeze(-1).transpose(1, 2)  # [batch_size, n_frames, n_features]
        x = self.gru1(x)[0]
        out_gru2 = self.gru2(x)[0]
        x = self.gru3(out_gru2)[0]
        x = x + out_gru2 # skip connection
        z = self.dense(x).unsqueeze(2)  # [batch_size, n_frames, 1 embedding_size]
        z = z.repeat(1, 1, self.n_sources, 1)  # [batch_size, n_frames, n_sources embedding_size]
        return z




# -------------------- Decoders ----------------------------------------



class SynthParameterDecoderSimple(torch.nn.Module):

    def __init__(self,
                 hidden_size=512,
                 output_size=512,
                 z_size=128,
                 bidirectional=True):
        super().__init__()

        if bidirectional: input_size = hidden_size * 2
        else: input_size = hidden_size

        self.f0_mlp = fc_stack(ch_in=1, ch_out=hidden_size, layers=3)
        self.z_mlp = fc_stack(ch_in=z_size, ch_out=hidden_size, layers=3)
        self.gru = torch.nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, batch_first=True, bidirectional=bidirectional)
        self.mlp = fc_stack(ch_in=input_size, ch_out=output_size, layers=3)


    def forward(self, f0_hz, z):
        """
        Args:
            f0_hz: torch.Tensor [batch_size, n_frames, 1]
            z: torch.Tensor [batch_size, n_frames, n_features]

        Returns:

        """
        batch_size, n_f0_frames, _ = f0_hz.shape
        batch_size, n_z_frames, n_z_features = z.shape

        # resample f0_hz if not same resolution as z
        if n_f0_frames != n_z_features:
            f0_hz = core.resample(f0_hz, n_z_frames)

        f0_unit = core.hz_to_unit(f0_hz)
        f0_latent = self.f0_mlp(f0_unit)
        z = self.z_mlp(z)
        z = torch.cat([f0_latent, z], dim=-1)
        z = self.gru(z)[0]
        z = self.mlp(z)
        return z



if __name__ == '__main__':


    encoder = SeparationEncoderWithF03()
    x = torch.rand(16, 64000)
    f0 = torch.ones((16, 250, 2)) * 300
    f0[:, :, 0] *= 2

    out = encoder(x, f0)
    print(out.shape)
    # decoder = SourceFilterFIRDecoder()
    # z = torch.rand(16, 250, 128)
    # f0 = torch.rand(16, 125, 1)
    # out = decoder(f0, z)
    # print(out.shape)

