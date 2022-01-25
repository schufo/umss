import argparse
import models
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import numpy as np
import random
import os
import copy
import configargparse
import shutil

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import model_utls
from ddsp import losses, spectral_ops

tqdm.monitor_interval = 0

def train(args, network, device, train_sampler, optimizer, ss_weights_dict):
    loss_container = utils.AverageMeter()
    network.train()

    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for data in pbar:
        pbar.set_description("Training batch")

        x = data[0]  # mix
        f0 = data[1]  # f0 [batch_size, n_frames, n_sources]
        targets = data[2]
        batch_size, n_frames, n_sources = f0.shape

        # randomly choose the target source
        probabilities = torch.ones((n_sources,))
        source_indices = torch.multinomial(probabilities, num_samples=batch_size, replacement=True)

        f0 = f0[torch.arange(batch_size), :, source_indices].unsqueeze(2)
        targets = targets[torch.arange(batch_size), :, source_indices]

        overlap = 1 - args.nhop/args.nfft
        target_mags = spectral_ops.compute_mag(targets, args.nfft, overlap, pad_end=True, center=True)

        x, f0, target_mags = x.to(device), f0.to(device), target_mags.to(device)

        optimizer.zero_grad()

        y_hat = network((x, f0))[:, 0, :, :]

        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(y_hat, target_mags)

        loss.backward()
        optimizer.step()
        loss_container.update(loss.item(), f0.size(0))
    return loss_container.avg


def valid(args, network, device, valid_sampler):
    loss_container = utils.AverageMeter()
    network.eval()
    with torch.no_grad():
        for data in valid_sampler:
            x = data[0]  # audio
            f0 = data[1]  # f0
            targets = data[2]
            batch_size, n_frames, n_sources = f0.shape

            # choose the target source
            source_indices = [s for s in range(n_sources)] * batch_size
            source_indices = source_indices[:batch_size]

            f0 = f0[torch.arange(batch_size), :, source_indices].unsqueeze(2)
            targets = targets[torch.arange(batch_size), :, source_indices]

            overlap = 1 - args.nhop/args.nfft
            target_mags = spectral_ops.compute_mag(targets, args.nfft, overlap, pad_end=True, center=True)

            x, f0, target_mags = x.to(device), f0.to(device), target_mags.to(device)

            y_hat = network((x, f0))[:, 0, :, :]

            loss_fn = torch.nn.L1Loss()
            loss = loss_fn(y_hat, target_mags)

            loss_container.update(loss.item(), f0.size(0))
        return loss_container.avg


def get_statistics(args, dataset):

    # dataset is an instance of a torch.utils.data.Dataset class

    scaler = sklearn.preprocessing.StandardScaler()  # tool to compute mean and variance of data

    # define operation that computes magnitude spectrograms
    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=True)
    )
    # return a deep copy of dataset:
    # constructs a new compound object and recursively inserts copies of the objects found in the original
    dataset_scaler = copy.deepcopy(dataset)

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None  # no scaling of sources before mixing
    dataset_scaler.random_chunks = False  # no random chunking of tracks
    dataset_scaler.random_track_mix = False  # no random accompaniments for vocals
    dataset_scaler.random_interferer_mix = False
    dataset_scaler.seq_duration = None  # if None, the original whole track from musdb is loaded

    # make a progress bar:
    # returns an iterator which acts exactly like the original iterable,
    # but prints a dynamically updating progressbar every time a value is requested.
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)

    for ind in pbar:
        out = dataset_scaler[ind]  # x is mix and y is target source in time domain, z is text and ignored here
        x = out[0]
        y = out[1]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])  # X is mono magnitude spectrogram, ... means as many ':' as needed

        # X is spectrogram of one full track
        # at this point, X has shape (nb_frames, nb_samples, nb_channels, nb_bins) = (N, 1, 1, F)
        # nb_frames: time steps, nb_bins: frequency bands, nb_samples: batch size

        # online computation of mean and std on X for later scaling
        # after squeezing, X has shape (N, F)
        scaler.partial_fit(np.squeeze(X))  # np.squeeze: remove single-dimensional entries from the shape of an array

    # set inital input scaler values
    # scale_ and mean_ have shape (nb_bins,), standard deviation and mean are computed on each frequency band separately
    # if std of a frequency bin is smaller than m = 1e-4 * (max std of all freq. bins), set it to m
    std = np.maximum(   # maximum compares two arrays element wise and returns the maximum element wise
        scaler.scale_,
        1e-4*np.max(scaler.scale_)  # np.max = np.amax, it returns the max element of one array
    )
    return scaler.mean_, std


def main():

    parser = configargparse.ArgParser()
    parser.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    #parser = argparse.ArgumentParser(description='Training')

    # experiment tag which will determine output folder in trained models, tensorboard name, etc.
    parser.add_argument('--tag', type=str, default='test')

    # allow to pass a comment about the experiment
    parser.add_argument('--comment', type=str, help='comment about the experiment')

    args, _ = parser.parse_known_args()

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="musdb",
                        help='Name of the dataset.')
    parser.add_argument('--one-example', action='store_true', default=False,
                        help='overfit to one example of the training set')
    parser.add_argument('--one-batch', action='store_true', default=False,
                        help='overfit to one batch of the training set')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='if True, correlated sources are generated in parallel in synth. dataset')
    parser.add_argument('--one-song', action='store_true', default=False,
                        help='if True, only one song is used in BC dataset for training and validation')



#parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--output', type=str, default="trained_models/{}/".format(args.tag),
                        help='provide output path base folder name')

    parser.add_argument('--wst-model', type=str, help='Path to checkpoint folder for warmstart')

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--supervised', action='store_true', default=False)


    parser.add_argument('--reconstruction-loss-weight', type=float, default=1.0)
    parser.add_argument('--loss-nfft', type=int, action='store', nargs='*', default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument('--loss-mag-weight', type=float, default=1.0)
    parser.add_argument('--loss-logmag-weight', type=float, default=1.0)
    parser.add_argument('--loss-logmel-weight', type=float, default=0.0)
    parser.add_argument('--loss-delta-freq-weight', type=float, default=0.0)
    parser.add_argument('--loss-delta-time-weight', type=float, default=0.0)
    parser.add_argument('--loss-lsf-weight', type=float, default=0.0)
    parser.add_argument('--ss-loss-weight', type=float, default=0.0)
    parser.add_argument('--harmonic-amp-loss-weight', type=float, default=0.0)
    parser.add_argument('--f0-hz-loss-weight', type=float, default=0.0)
    parser.add_argument('--harmonics-roll-off-loss-weight', type=float, default=0.0)
    parser.add_argument('--lsf-loss-weight', type=float, default=0.0)
    parser.add_argument('--noise-gain-loss-weight', type=float, default=0.0)
    parser.add_argument('--noise-mags-loss-weight', type=float, default=0.0)



    # Model Parameters
    parser.add_argument('--nfft', type=int, default=512,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=256,
                        help='STFT hop size')
    parser.add_argument('--filter-order', type=int, default=10,
                        help='filter order of vocal tract all-pole filter')

    parser.add_argument('--noise-filter-mags', type=int, default=40,
                        help='number of frequency bands in noise filter')
    parser.add_argument('--encoder', type=str, default='SeparationEncoderSimple')
    parser.add_argument('--encoder-hidden-size', type=int, default=256)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--decoder-hidden-size', type=int, default=512)
    parser.add_argument('--decoder-output-size', type=int, default=512)
    parser.add_argument('--n-sources', type=int, default=2)
    parser.add_argument('--estimate-lsf', action='store_true', default=False)
    parser.add_argument('--estimate-noise-mags', action='store_true', default=False)
    parser.add_argument('--unidirectional', action='store_true', default=False)
    parser.add_argument('--voiced-unvoiced-same-noise', action='store_true', default=False)


    parser.add_argument('--nb-workers', type=int, default=4,
                        help='Number of workers for dataloader.')

    # name of the model class in model.py that should be used
    parser.add_argument('--original-cu-net', action='store_true', default=False)

    parser.add_argument('--architecture', type=str)
    parser.add_argument('--nb-filter-magnitudes', type=int, default=65)
    parser.add_argument('--estimate-f0', action='store_true', default=False)
    parser.add_argument('--supervised-f0', action='store_true', default=False)
    parser.add_argument('--switch-off-noise', action='store_true', default=False)
    parser.add_argument('--f-ref-source-spec', type=float, default=200.)
    parser.add_argument('--harmonic-roll-off', type=float, default=12.)
    parser.add_argument('--source-spectrum', type=str, default='flat')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    #print("Using Torchaudio: ", utils._torchaudio_available())
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    # copy config.txt to output dir
    shutil.copy2('unet_config.txt', target_path)

    writer = SummaryWriter(log_dir=os.path.join('tensorboard', args.tag))

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)


    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        worker_init_fn=utils.worker_init_fn,
        **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, **dataloader_kwargs
    )

    # make dict for self supervision loss weights
    ss_weights_dict = {'harmonic_amplitudes': args.harmonic_amp_loss_weight,
                       'harmonic_distribution': 0.,
                       'f0_hz': args.f0_hz_loss_weight,
                       'harmonics_roll_off': args.harmonics_roll_off_loss_weight,
                       'line_spectral_frequencies': args.lsf_loss_weight,
                       'noise_gain': args.noise_gain_loss_weight,
                       'voiced_unvoiced': 0.,
                       'voiced_noise_magnitudes': args.noise_mags_loss_weight,
                       }


    train_args_dict = vars(args)

    train_params_dict = copy.deepcopy(vars(args))  # return args as dictionary with no influence on args


    model_class = model_utls.ModelLoader.get_model(args.architecture)
    model_to_train = model_class.from_config(train_params_dict)
    model_to_train.to(device)

    optimizer = torch.optim.Adam(
        model_to_train.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=args.lr_decay_gamma)

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.wst_model:
        model_path = Path(os.path.join('trained_models', args.wst_model)).expanduser()
        with open(Path(model_path, args.wst_model + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.wst_model + ".pth")
        state_dict = torch.load(target_model_path, map_location=device)
        model_to_train.load_state_dict(state_dict)

        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        # we don't set the parameters below to allow resuming training on different data set
        # (model is saved with new name, so there is no risk of overwriting)
        best_epoch = 0
        #es.best = results['best_loss']
        #es.num_bad_epochs = results['num_bad_epochs']
    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()

        train_loss = train(args, model_to_train, device, train_sampler, optimizer, ss_weights_dict)

        # calculate validation loss only if model is not optimized on one single example
        if args.one_example or args.one_batch or (args.dataset == 'synthetic'):
            # if overfitting on one example, early stopping is done based on training loss
            valid_loss = train_loss
        else:
            valid_loss = valid(args, model_to_train, device, valid_sampler)
            writer.add_scalar("Validation_cost", valid_loss, epoch)
            valid_losses.append(valid_loss)

        writer.add_scalar("Training_cost", train_loss, epoch)

        scheduler.step()
        train_losses.append(train_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_train.state_dict(),
            'best_loss': es.best,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
            is_best=valid_loss == es.best,
            path=target_path,
            tag=args.tag
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs
        }

        with open(Path(target_path,  args.tag + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
