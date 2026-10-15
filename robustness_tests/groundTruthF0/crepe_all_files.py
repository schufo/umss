'''This function uses CREPE (via torchcrepe) to individually evaluate pitch for each of the voices commonly used during evaluations.

Writes one <name>.f0.csv per voice in the format of the original crepe command line tool
(crepe <file> --step-size 16), i.e. columns time, frequency, confidence.
The committed csv files in crepe_centered/ were produced with the original TensorFlow crepe;
by default this script writes to torchcrepe_centered/ so they are not overwritten.

Run from robustness_tests/groundTruthF0: python crepe_all_files.py [--out-dir DIR]
'''

import argparse
import os

import numpy as np
import soundfile
import torch
import torchcrepe

tags = [
    'El_Rossinyol/audio_16kHz/rossinyol_Bajos_207.wav',
	'El_Rossinyol/audio_16kHz/rossinyol_ContraAlt_2-06.wav',
	'El_Rossinyol/audio_16kHz/rossinyol_Soprano_208.wav',
	'El_Rossinyol/audio_16kHz/rossinyol_Tenor2-09.wav',
    'Locus_Iste/audio_16kHz/locus_Bajos_3-02.wav',
	'Locus_Iste/audio_16kHz/locus_ContraAlt_301.wav',
	'Locus_Iste/audio_16kHz/locus_Soprano_310.wav',
	'Locus_Iste/audio_16kHz/locus_tenor3-01-2.wav',
    'Nino_Dios/audio_16kHz/nino_Bajos_404.wav',
	'Nino_Dios/audio_16kHz/nino_ContraAlt_407.wav',
	'Nino_Dios/audio_16kHz/nino_Soprano_405.wav',
	'Nino_Dios/audio_16kHz/nino_tenor4-06-2.wav',
        ]

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, default='torchcrepe_centered')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
step_size_ms = 16

for tag in tags:
	path = '../../Datasets/ChoralSingingDataset/{}'.format(tag)
	print('torchcrepe', path, '--step-size', step_size_ms)
	audio, sr = soundfile.read(path, dtype='float32', always_2d=True)
	audio = torch.from_numpy(audio.mean(axis=1))[None, :]
	# settings of the crepe command line tool: full model, centered frames, no viterbi decoding
	frequency, confidence = torchcrepe.predict(audio, sr, hop_length=int(sr * step_size_ms / 1000),
											   fmin=32.70, fmax=1975.5, model='full',
											   decoder=torchcrepe.decode.weighted_argmax,
											   return_periodicity=True, pad=True, batch_size=512, device=device)
	frequency, confidence = frequency[0].cpu().numpy(), confidence[0].cpu().numpy()
	time = np.arange(len(frequency)) * step_size_ms / 1000
	out_file = os.path.join(args.out_dir, os.path.basename(tag).replace('.wav', '.f0.csv'))
	np.savetxt(out_file, np.stack([time, frequency, confidence], axis=1), fmt=['%.3f', '%.3f', '%.6f'],
			   delimiter=',', header='time,frequency,confidence', comments='')
#the --no-centering flag isn't appropriate!
