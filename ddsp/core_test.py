import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt
import spectrum

import unittest

from ddsp import core, synths


class TestCore(unittest.TestCase):

    def test_fft_convolve_is_accurate(self):
        """Tests convolving signals using fast fourier transform (fft).
        Generate random signals and convolve using fft. Compare outputs to the
        implementation in scipy.signal.

        """
        # Create random signals to convolve.
        audio = np.ones([1, 1000]).astype(np.float32)
        impulse_response = np.ones([1, 50]).astype(np.float32)

        output_pt = core.fft_convolve(
            torch.from_numpy(audio), torch.from_numpy(impulse_response), padding='valid', delay_compensation=0)[0]

        output_np = signal.fftconvolve(audio[0], impulse_response[0])

        difference = output_np - output_pt.numpy()
        total_difference = np.abs(difference).mean()
        # print(total_difference)

        threshold = 1e-3
        self.assertLessEqual(total_difference, threshold)


    def test_reflection_to_filter_coeff(self):

        """
        frame_1: k_1 = 0.5, k_2 = 0.1, k_3 = 0.3  (frame_2: k_1 = 0.4, k_2 = 0.1, k_3 = 0.3)

        i = 1:
            a_1^(1) = k_1 = 0.5  (0.4)

        i = 2:
            a_2^(2) = k_2 = 0.1
                j = 1: a_1^(2) = a_1^(1) - k_2*a_1^(1) = 0.5 - 0.1*0.5 = 0.45  (0.4 - 0.1*0.4 = 0.36)

        i = 3:
            a_3^(3) = k_3 = 0.3
                j = 1: a_1^(3) = a_1^(2) - k_3*a_2^(2) = 0.45 - 0.3*0.1  =  0.42  (0.36 - 0.3*0.1 = 0.33)
                j = 2: a_2^(3) = a_2^(2) - k_3*a_1^(2) = 0.1  - 0.3*0.45 = -0.035 (0.1 - 0.3*0.36 = -0.008)

        """

        reflection_coeff = torch.zeros((2, 2, 3))  # [batch_size, n_frames, n_coeff]
        reflection_coeff[:, 0, 0] = 0.5
        reflection_coeff[:, 0, 1] = 0.1
        reflection_coeff[:, 0, 2] = 0.3
        reflection_coeff[:, 1, 0] = 0.4
        reflection_coeff[:, 1, 1] = 0.1
        reflection_coeff[:, 1, 2] = 0.3

        filter_coeff_expected = torch.tensor([[[0.42, -0.035, 0.3], [0.33, -0.008, 0.3]],
                                              [[0.42, -0.035, 0.3], [0.33, -0.008, 0.3]]]).numpy()
        filter_coeff_computed = core.reflection_to_filter_coeff(reflection_coeff).numpy()

        for i in range(3):
            self.assertAlmostEqual(filter_coeff_expected[0, 0, i], filter_coeff_computed[0, 0, i])
            self.assertAlmostEqual(filter_coeff_expected[1, 0, i], filter_coeff_computed[1, 0, i])
            self.assertAlmostEqual(filter_coeff_expected[0, 1, i], filter_coeff_computed[0, 1, i])
            self.assertAlmostEqual(filter_coeff_expected[1, 1, i], filter_coeff_computed[1, 1, i])


    def test_apply_all_pole_filter(self):
        """
        Test if core.apply_all_pole_filter returns the same filtered signal as scipy.signal.lfilter
        when given the same filter coefficients. This tests the case of a time-invariant filter.
        The filter coefficients are copied in 200 blocks to test the overlap add procedure in
        the function.
        """

        # generate a source signal that will be filtered
        harmonic_synth = synths.Harmonic(n_samples=64000, sample_rate=16000)
        amplitudes = torch.ones((1, 250, 1))  # [batch, n_frames, 1]
        harmonic_distribution = torch.ones((1, 250, 15))  # [batch, n_frames, n_harmonics]
        f0 = torch.ones((1, 250, 1)) * torch.linspace(200, 200, 250)[None, :, None]  # [batch, n_frames, 1]
        source_signal = harmonic_synth(amplitudes, harmonic_distribution, f0)  # [batch_size, 64000]

        # generate filter coefficients that will lead to a stable filter
        b, a = signal.iirfilter(N=5, Wn=0.2, btype='lowpass', output='ba')

        # make all-pole filter by setting b_0 = 1 and b_i = 0 for all i
        b = np.zeros_like(a)
        b[0] = 1.

        y_scipy = signal.lfilter(b=b, a=a, x=source_signal.numpy()[0, :])

        a_torch = torch.tensor(a[1:])[None, None, :]  # remove a_0 and add batch and frame dimensions
        a_torch = torch.cat([a_torch] * 200, dim=1)  # make 200 frames with the same filter coefficients

        audio_block_length = int((64000 / 200) * 2)  # 200 blocks with 50 % overlap --> length=640
        y_test = core.apply_all_pole_filter(source_signal, a_torch, audio_block_size=audio_block_length, parallel=True)
        y_test = y_test[0, :64000].numpy()

        difference = y_scipy - y_test
        total_difference = np.abs(difference).mean()

        threshold = 1e-3
        self.assertLessEqual(total_difference, threshold)


    def test_lsf_to_filter_coeff(self):

        lsf = [0.0483, 0.1020, 0.1240, 0.2139, 0.3012, 0.5279, 0.6416, 0.6953, 0.9224,
               1.1515, 1.2545, 1.3581, 1.4875, 1.7679, 1.9860, 2.2033, 2.3631, 2.5655,
               2.6630, 2.8564]

        a_pyspec = spectrum.lsf2poly(lsf)  # numpy-based method
        a_torch = core.lsf_to_filter_coeff(torch.tensor(lsf)[None, None, :])  # PyTorch implementation

        a_pyspec = a_pyspec[1:]  # remove 0th coefficient a_0 = 1
        a_torch = a_torch.numpy()[0, 0, :]

        difference = a_pyspec - a_torch
        mean_difference = abs(difference).mean()

        threshold = 1e-5
        self.assertLessEqual(mean_difference, threshold)


if __name__ == '__main__':

    # test = TestCore()
    # test.test_apply_all_pole_filter()
    # quit()

    unittest.main()