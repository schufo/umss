"""
This script goes through f0 estimates of all sources from mixtures produced with
the model of Cuesta et al. (ISMIR 2020). Then the estimates are allocated to the respective sources
using simple continuity rules, resampled and saved as torch tensors.
"""
import csv
import itertools
import glob
import os

import numpy as np
import torch

import ddsp.core
import data


def compute_frame_distance(frame_1, frame_2, assigned_frames=None, n=0, backward_pass=False):
    """

    Args:
        frame_1: list of f0 values. This frame is considered as the one whose configuration is being tested.
            Its entries which are zero and the corresponding values in frame_2 will be ignored for the
            distance computation.
        frame_2: list of f0 values. This frame is considered to be a reference frame to enable a decision
            about frame_1. Its entries which are zero (and are not ignored due to zeros in frame_1) will
            be replaced by the first previous/subsequent non-zero entry of assigned_frames. The decision whether
            to look for previous or subsequent non-zero entries is taken based on the argument 'backward_pass'.
        assigned_frames: np.array of shape [n_frames, n_sources] containing already assigned f0 values and zeros
            as placeholders for frames that are not assigned yet.
        n: int, frame index of frame_1 in assigned_frames
        backward_pass: If True, it is assumed that the distance between frame_1 and frame_2 is computed in a
            backward pass through assigned_frames and therefor a zero in frame_2 will be replaced by a subsequent
            non-zero value (with frame index > n). If False, a previous non-zero value (frame index < n) is taken.

    Returns:
        distance: float, the distance between the two frames.

    """

    frame_2 = frame_2.copy()
    assigned_frames = assigned_frames.copy()

    non_zero_idx_1 = np.nonzero(frame_1)[0]
    frame_1_entries = np.array(frame_1)[non_zero_idx_1]
    frame_2_entries = np.array(frame_2)[non_zero_idx_1]

    if not np.all(frame_2_entries):
        # there are still zeros in frame_2_entries, replace them by f0 values of previous/subsequent frames
        if backward_pass: assigned_frames = assigned_frames[::-1, :]  # reverse assigned frames
        for s, f0_value in enumerate(frame_2):
            if f0_value > 0: continue
            m = -n - 1 + assigned_frames.shape[0] if backward_pass else n
            x = 0
            while x == 0:
                x = assigned_frames[m-2, s]
                m -= 1
                if m < 2: break
            frame_2[s] = x
        frame_2_entries = np.array(frame_2)[non_zero_idx_1]

    distance = sum(abs(frame_1_entries - frame_2_entries))
    return distance



path_to_dataset = '../Datasets/ChoralSingingDataset'
songs = ['El Rossinyol', 'Locus Iste', 'Nino Dios']

mixture_dirs = ['mixtures_2_sources', 'mixtures_3_sources', 'mixtures_4_sources']

for song in songs:
    for s, mix_dir in enumerate(mixture_dirs):

        n_sources = s + 2
        print(n_sources)

        path_to_f0_csv_files = os.path.join(path_to_dataset, song, mix_dir)
        path_to_save_f0_estimate_tensors = os.path.join(path_to_f0_csv_files, 'mf0_cuesta_processed')

        # make directory to save processed f0 estimates as torch tensor
        if not os.path.isdir(path_to_save_f0_estimate_tensors):
            os.makedirs(path_to_save_f0_estimate_tensors, exist_ok=True)

        f0_csv_files = sorted(list(glob.glob(path_to_f0_csv_files + '/*.csv')))

        for f0_csv_file in f0_csv_files:
            time = []
            f0_estimates = []
            with open(f0_csv_file, newline='') as csv_file:
                reader = csv.reader(csv_file, delimiter='\t', quotechar='|')
                for row in reader:
                    row = [float(x) for x in row]
                    while len(row) < n_sources + 1: row.append(0)  # fill frames without detected f0s with zeros
                    time.append(row[0])
                    f0_estimates.append(sorted(row[1:]))  # assume that the voices do not cross in f0 and assign ordered f0 to ordered sources

            labels = []
            for n, freqs in enumerate(f0_estimates):
                if len(freqs) > n_sources: labels.append('n_f0>n_s')
                elif sum(freqs) == 0: labels.append('all_zero')
                elif freqs[0] == 0 and sum(freqs[1:]) != 0: labels.append('1+_zero')
                else: labels.append('no_zero')

            idx_decision_required = [i for i, j in enumerate(labels) if j == '1+_zero' or j == 'n_f0>n_s']

            subsequence_start = []
            subsequence_end = []
            labels_that_require_decision = ['1+_zero', 'n_f0>n_s']
            for n, label in enumerate(labels):

                if n < len(labels) - 1 and label not in labels_that_require_decision and labels[n+1] in labels_that_require_decision:
                    subsequence_start.append((n, label))
                if n > 0 and label not in labels_that_require_decision and labels[n-1] in labels_that_require_decision:
                    subsequence_end.append((n, label))

            assert len(subsequence_start) == len(subsequence_end), 'these lists must have the same length'

            f0_assigned = np.zeros((len(f0_estimates), n_sources))

            # assign f0 values that do not need any decision (they are assigned by sorting)
            for n, f0s in enumerate(f0_estimates):
                if n not in idx_decision_required:
                    f0_assigned[n, :] = f0s

            # make decisions for the rest of the f0 estimates
            for n, start_boundary in enumerate(subsequence_start):
                start_boundary_frame, start_label = start_boundary
                end_boundary_frame, end_label = subsequence_end[n]

                if start_label == 'no_zero' and end_label == 'all_zero':
                    # forward pass
                    for m in range(start_boundary_frame+1, end_boundary_frame):
                        permutations = list(itertools.permutations(f0_estimates[m]))
                        distance_to_prev = []
                        for p in permutations:
                            p = p[:n_sources]
                            distance = compute_frame_distance(p, f0_assigned[m-1, :], f0_assigned, n=m, backward_pass=False)
                            distance_to_prev.append(distance)
                        best_perm_idx = np.argmin(distance_to_prev)
                        f0_assigned[m, :] = permutations[best_perm_idx][:n_sources]

                elif start_label == 'all_zero' and end_label == 'no_zero':
                    # backward pass
                    for m in range(end_boundary_frame - 1, start_boundary_frame, -1):
                        permutations = list(itertools.permutations(f0_estimates[m]))
                        distance_to_prev = []
                        for p in permutations:
                            p = p[:n_sources]
                            distance = compute_frame_distance(p, f0_assigned[m+1, :], f0_assigned, n=m, backward_pass=True)
                            distance_to_prev.append(distance)
                        best_perm_idx = np.argmin(distance_to_prev)
                        f0_assigned[m, :] = permutations[best_perm_idx][:n_sources]

                elif start_label == end_label == 'no_zero':
                    # forward and backward pass

                    # forward pass
                    forward_distances = []
                    forward_f0_assignments = []
                    for m in range(start_boundary_frame+1, end_boundary_frame):
                        permutations = list(itertools.permutations(f0_estimates[m]))
                        distance_to_prev = []
                        for p in permutations:
                            p = p[:n_sources]
                            distance = compute_frame_distance(p, f0_assigned[m-1, :], f0_assigned, n=m, backward_pass=False)
                            distance_to_prev.append(distance)
                        best_perm_idx = np.argmin(distance_to_prev)
                        forward_distances.append(min(distance_to_prev))
                        forward_f0_assignments.append(permutations[best_perm_idx][:n_sources])
                        f0_assigned[m, :] = permutations[best_perm_idx][:n_sources]
                    forward_distances.append(compute_frame_distance(permutations[best_perm_idx][:n_sources], f0_assigned[end_boundary_frame, :], f0_assigned))
                    forward_distance = sum(forward_distances)
                    f0_assigned[start_boundary_frame+1:end_boundary_frame, :] = [0] * n_sources

                    # backward pass
                    backward_distances = []
                    backward_f0_assignments =[]
                    for m in range(end_boundary_frame - 1, start_boundary_frame, -1):
                        permutations = list(itertools.permutations(f0_estimates[m]))
                        distance_to_prev = []
                        for p in permutations:
                            p = p[:n_sources]
                            distance = compute_frame_distance(p, f0_assigned[m+1, :], f0_assigned, n=m, backward_pass=True)
                            distance_to_prev.append(distance)
                        best_perm_idx = np.argmin(distance_to_prev)
                        backward_distances.append(min(distance_to_prev))
                        backward_f0_assignments.append(permutations[best_perm_idx][:n_sources])
                        f0_assigned[m, :] = permutations[best_perm_idx][:n_sources]
                    backward_distances.append(compute_frame_distance(permutations[best_perm_idx][:n_sources], f0_assigned[start_boundary_frame, :], f0_assigned))
                    backward_distance = sum(backward_distances)

                    # print('forward_distances', forward_distances)
                    # print('backward_distances', backward_distances)
                    # compare forward and backward accumulated distances and decide which assignment to take
                    if forward_distance <= backward_distance:
                        f0_assigned[start_boundary_frame+1:end_boundary_frame, :] = forward_f0_assignments

                elif start_label == end_label == 'all_zero':
                    # find first previous/subsequent no_zero frames for start/end frames and use them as boundaries
                    try: prev_no_zero_frame = start_boundary_frame - 1 - labels[:start_boundary_frame][::-1].index('no_zero')
                    except ValueError: prev_no_zero_frame = start_boundary_frame
                    try: next_no_zero_frame = end_boundary_frame + labels[end_boundary_frame:].index('no_zero')
                    except ValueError: next_no_zero_frame = end_boundary_frame

                    # forward pass
                    forward_distances = []
                    forward_f0_assignments = []
                    for m in range(prev_no_zero_frame+1, next_no_zero_frame):
                        permutations = list(itertools.permutations(f0_estimates[m]))
                        distance_to_prev = []
                        for p in permutations:
                            p = p[:n_sources]
                            distance = compute_frame_distance(p, f0_assigned[m-1, :], f0_assigned, n=m, backward_pass=False)
                            distance_to_prev.append(distance)
                        best_perm_idx = np.argmin(distance_to_prev)
                        forward_distances.append(min(distance_to_prev))
                        forward_f0_assignments.append(permutations[best_perm_idx][:n_sources])
                        f0_assigned[m, :] = permutations[best_perm_idx][:n_sources]
                    forward_distance = sum(forward_distances)
                    f0_assigned[prev_no_zero_frame+1:next_no_zero_frame, :] = [0] * n_sources

                    # backward pass
                    backward_distances = []
                    backward_f0_assignments =[]
                    for m in range(next_no_zero_frame - 1, prev_no_zero_frame, -1):
                        permutations = list(itertools.permutations(f0_estimates[m]))
                        distance_to_prev = []
                        for p in permutations:
                            p = p[:n_sources]
                            distance = compute_frame_distance(p, f0_assigned[m+1, :], f0_assigned, n=m, backward_pass=True)
                            distance_to_prev.append(distance)
                        best_perm_idx = np.argmin(distance_to_prev)
                        backward_distances.append(min(distance_to_prev))
                        backward_f0_assignments.append(permutations[best_perm_idx][:n_sources])
                        f0_assigned[m, :] = permutations[best_perm_idx][:n_sources]
                    backward_distance = sum(backward_distances)

                    # print('forward_distances', forward_distances)
                    # print('backward_distances', backward_distances)
                    # compare forward and backward accumulated distances and decide which assignment to take
                    if forward_distance <= backward_distance:
                        f0_assigned[prev_no_zero_frame+1:next_no_zero_frame, :] = forward_f0_assignments

                else:
                    # one label is 'n_f0>n_s' and needs to be corrected before making a decision here
                    pass


            if song == 'El Rossinyol': audio_length = 134
            elif song == 'Locus Iste': audio_length = 190
            elif song == 'Nino Dios': audio_length = 103

            # compute number of STFT frames for the song
            n_stft_frames = 16000 * audio_length // 256

            # resample and save as torch.Tensor
            f0_cuesta = torch.tensor(f0_assigned).transpose(0, 1)  # [n_sources, n_frames]
            f0_cuesta = ddsp.core.resample(f0_cuesta, n_stft_frames)
            f0_cuesta = f0_cuesta.transpose(0, 1)  # [n_frames, n_sources]
            f0_cuesta = f0_cuesta.flip(dims=(1,))

            # save as torch tensor
            name = f0_csv_file.split('/')[-1][:-4]
            print(name)
            torch.save(f0_cuesta, os.path.join(path_to_save_f0_estimate_tensors, name + '.pt'))