'''This function uses CREPE to individually evaluate pitch for each of the voices commonly used during evaluations.'''

import os

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

for tag in tags:
	print("crepe ../../Dataset/ChoralSingingDataset/{} --step-size 16".format(tag))
	os.system("crepe ../../Dataset/ChoralSingingDataset/{} --step-size 16".format(tag))
#the --no-centering flag isn't appropriate!
