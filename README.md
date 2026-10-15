# Unsupervised Music Source Separation Using Differentiable Parametric Source Models

This is the source code for the experiments related to the paper [Unsupervised Music Source Separation Using Differentiable Parametric Source Models](https://arxiv.org/abs/2201.09592).  

It contains a re-implementation of parts of the DDSP library in PyTorch. We added a differentiable all-pole filter which can be parameterized by line spectral frequencies or reflection coefficients. 

Please cite the paper, if you use parts of the code in your work.

## Links
[:loud_sound: Audio examples](https://schufo.github.io/umss/)

[:page_facing_up: Paper](https://arxiv.org/abs/2201.09592)

[:file_folder: CSD Database](https://zenodo.org/record/1286570#.Y0ZsbNJByUk)

[:microphone: Multiple-f0 estimation](https://github.com/helenacuesta/multif0-estimation-polyvocals)

## Requirements

Python 3.12 or newer. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
Install [uv](https://docs.astral.sh/uv/), then from the repository root:

    uv sync

This creates a `.venv` with PyTorch, torchaudio, librosa, etc. On Linux x86_64 the default PyTorch wheel includes CUDA support.
To also install the dependencies of the robustness tests (`torchcrepe`, `mido`):

    uv sync --extra robustness

Run any script with `uv run python <script>.py ...` (or activate `.venv` first).

The scripts use a CUDA GPU if available, otherwise Apple Silicon (MPS), otherwise the CPU.

## Training

    uv run python train.py -c config.txt

    uv run python train_u_nets.py -c unet_config.txt

Pass `--no-cuda` to train on the CPU even when a GPU is available.

## Evaluation

    uv run python eval.py --tag 'TAG' --f0-from-mix --test-set 'CSD'

Note : 'TAG' is the evaluated model's name. (Example: unsupervised_2s_satb_bcbq_mf0_1)
Use `--device cpu|cuda|mps` to choose the device explicitly.

## Tests

    uv run python -m unittest ddsp.core_test

## Notes on the dependency update

The code was originally written for PyTorch 1.6 / torchaudio 0.6 / librosa 0.8 and has been ported to current versions.
On the CPU, the ported DDSP operations (including gradients) and the evaluation metrics match the original
environment within float32 precision, and evaluating the provided models gives the same results.
Two intentional details keep the old behaviour:
`frequency_impulse_response` produces impulse responses of length `2 * n_frequencies - 1` like the removed `torch.irfft`,
and the librosa STFTs use `pad_mode='reflect'` (the default before librosa 0.10).
`robustness_tests/groundTruthF0/crepe_all_files.py` now uses torchcrepe instead of the TensorFlow crepe tool;
its pitch estimates differ slightly (a few cents) from the committed `crepe_centered` files, which remain the reference.

## Acknowledgment

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

## Copyright

Copyright 2021 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris.
All rights reserved.
