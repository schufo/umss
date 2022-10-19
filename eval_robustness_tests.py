'''
The main objective of this evaluation function is to run many evaluations using Ground Truth F0 instead
of multi-pitch f0 mixtures, while adding varying amounts of different errors to observe their influence
on the evaluation results.

To achieve this, we get rid of 'f0_hz = d[1].to(device)' from the original eval.py function, and
we replace it with our own crepe ground truth files.
We have to make sure our f0_hz ground truth is identical in format to the d[1] from data.py (a CSD item).

Since for the paper we were mostly interested in SI-SDR : For faster evals, only it is being calculated.

Results are saved to evaluation folders and to a csv log file.
If an evaluation folder corresponding to a test already exists, the test will skipped.
The log file will not clean itself if you delete evaluation folders.

As of right now, having only one seed is forced, for no reason other than for speed.
As of right now, for the n_strict_errors test, there is only one mask seed.
As of right now, parsed argument --show-progress is not implemented, rather a ugly progress bar was implemented.

This function only works with models estimating 4 sources.
'''

#from cmath import nan
#from operator import index
import os
import pickle
import json
import argparse

import torch
from numpy import NaN, float32
import numpy as np
import pandas as pd
import librosa as lb

import data
import models
import utils
import evaluation_metrics as em
import ddsp.spectral_ops

import math
import random

# Manual option
force_only_one_seed=True

# Random seed initialization
torch.manual_seed(0)
random.seed(0)
random_state=random.getstate()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--test-set', type=str, default='El Rossinyol', choices=['CSD'])
parser.add_argument('--f0-from-mix', action='store_true', default=True)
args, _ = parser.parse_known_args()
tag = args.tag
is_u_net = tag[:4] == 'unet'
parser.add_argument('--eval-tag', type=str, default=tag)
args, _ = parser.parse_known_args()
f0_cuesta = args.f0_from_mix

parser.add_argument('--teststocompute', nargs='+', default=['all'], choices=['all','baseline_gtf0','gtf0_transposed',
                                                        'gtf0_octaved','gtf0_voices_missing','gtf0_strict_error_percent'])
a=0 # this is an awful way to do this
for _ , value in parser.parse_args()._get_kwargs():
    if a<=3:
        a+=1
        continue
    tests_to_compute=value
    if 'all' in tests_to_compute:
        tests_to_compute=['baseline_gtf0','gtf0_transposed','gtf0_octaved',
                            'gtf0_voices_missing','gtf0_strict_error_percent']
    break

# Load model arguments
device = 'cuda' if torch.cuda.is_available() else 'cpu' #Cuda is mandatory to reproduce results from the paper.
trained_model, model_args = utils.load_model(tag, device, return_args=True)
trained_model.return_synth_params = False
trained_model.return_sources=True
voices = model_args['voices'] if 'voices' in model_args.keys() else 'satb'
original_cunet = model_args['original_cu_net'] if 'original_cu_net' in model_args.keys() else False

# Initialize unique results path.
if args.f0_from_mix: f0_add_on = 'mf0'
if args.test_set == 'CSD': test_set_add_on = 'CSD'
path_to_save_results_masking = 'robustness_tests/evaluation/{}/eval_results_{}_{}_{}'.format(args.eval_tag , f0_add_on, test_set_add_on, device)
if not os.path.isdir(path_to_save_results_masking):
    os.makedirs(path_to_save_results_masking, exist_ok=True)

# Initialize test_set
if args.test_set == 'CSD':
    el_rossinyol = data.CSD(song_name='El Rossinyol', example_length=model_args['example_length'], allowed_voices=voices,return_name=True, n_sources=model_args['n_sources'], singer_nb=[2], random_mixes=False,f0_from_mix=f0_cuesta, cunet_original=original_cunet)
    locus_iste = data.CSD(song_name='Locus Iste', example_length=model_args['example_length'], allowed_voices=voices,return_name=True, n_sources=model_args['n_sources'], singer_nb=[3], random_mixes=False,f0_from_mix=f0_cuesta, cunet_original=original_cunet)
    nino_dios = data.CSD(song_name='Nino Dios', example_length=model_args['example_length'], allowed_voices=voices,return_name=True, n_sources=model_args['n_sources'], singer_nb=[4], random_mixes=False,f0_from_mix=f0_cuesta, cunet_original=original_cunet)
    test_set = torch.utils.data.ConcatDataset([el_rossinyol, locus_iste, nino_dios])

pd.set_option("display.max_rows", None, "display.max_columns", None)

def find_nth(haystack,needle, n):
    '''Useful function to find the nth specific character in as string'''
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle,start+len(needle))
        n -=1
    return start

def load_groundTruthF0_files(gtf0_source="crepe_centered"):
    '''
    gtf0_source : midi, crepe_centered, crepe_no-centering
    returns a dictionnary with the 3 dataframes corresponding to each song.
    '''
    #TODO Could one directly torch.load these files?
    #f0_estimates = torch.load(f0_from_mix_file)[crepe_start_frame:crepe_end_frame, :]
    el_rossinyol_gtf0_df=pd.read_csv('./robustness_tests/groundTruthF0/{}/El_Rossinyol_SATB_gtf0.csv'.format(gtf0_source))
    locus_iste_gtf0_df=pd.read_csv('./robustness_tests/groundTruthF0/{}/Locus_Iste_SATB_gtf0.csv'.format(gtf0_source))
    nino_dios_gtf0_df=pd.read_csv('./robustness_tests/groundTruthF0/{}/Nino_Dios_SATB_gtf0.csv'.format(gtf0_source))
    if gtf0_source=="midi" :
        el_rossinyol_gtf0_df.index=el_rossinyol_gtf0_df.pop('time')
        locus_iste_gtf0_df.index=locus_iste_gtf0_df.pop('time')
        nino_dios_gtf0_df.index=nino_dios_gtf0_df.pop('time')
    else :
        el_rossinyol_gtf0_df.index=el_rossinyol_gtf0_df.pop('time').astype({'time':int}, errors='ignore')
        locus_iste_gtf0_df.index=locus_iste_gtf0_df.pop('time').astype({'time':int}, errors='ignore')
        nino_dios_gtf0_df.index=nino_dios_gtf0_df.pop('time').astype({'time':int}, errors='ignore')
    song_gtf0_dict={'el_rossinyol':el_rossinyol_gtf0_df,
                    'locus_iste':locus_iste_gtf0_df,
                    'nino_dios':nino_dios_gtf0_df}
    return(song_gtf0_dict)

def eval_body(n_seeds,test_set,el_rossinyol_gtf0_df,locus_iste_gtf0_df,nino_dios_gtf0_df):
    eval_results_masking = pd.DataFrame({'mix_name': [], 'eval_seed': [], 'voice': [], 'eval_frame': [],'SI-SDR': []})
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        rng_state_torch = torch.get_rng_state()

        for idx in range(len(test_set)): #for every 4 second CSD item. A batch is 4x1 second.
            d = test_set[idx] #csd get_item

            name = d[3]
            voices = d[4]

            print('seed :',seed, '| completion :', idx,'/ 101', '| mixture name :', name)

            # Find current ms start point
            startOfItemWindow=math.ceil(float(name[-(len(name)-name.rfind("_")-1):])) #reads last floating point number from the mix_name which indicates the start of the CSD item Window and converts it to int
            startOfItemWindow=(1000*startOfItemWindow) # convert to ms
            
            # Find current song name
            current_song=name[:find_nth(name,'_',2)] #can be el_rossinyol, locus_iste, nino_dios

            # select correct ground truth f0 dataframe according to the current song
            which_df_for_song={
                'el_rossinyol':el_rossinyol_gtf0_df,
                'locus_iste':locus_iste_gtf0_df,
                'nino_dios':nino_dios_gtf0_df
                }
            gtf0=which_df_for_song[current_song]

            # localize the interesting part of gtf0 that will be fed to f0_hz
            d1_replacer=gtf0.iloc[[ startOfItemWindow//16 -1 + x for x in range(250) ]][['s','a','t','b']]
            d1_replacer = torch.tensor(d1_replacer.values.astype(float32))

            # load d1_replacer into f0_hz to eval as usual
            #Instead of using the Cuesta-generated F0 mixtures, using CREPE Ground Truth f0s # f0_hz = d[1].to(device) 
            f0_hz = d1_replacer.to(device)
            f0_hz = f0_hz[None, :, :] # formatting

            # evaluation as usual
            mix = d[0].to(device)
            target_sources = d[2].to(device)
            mix = mix.unsqueeze(0)
            target_sources = target_sources.unsqueeze(0)
            batch_size, n_samples, n_sources = target_sources.shape
            n_fft_metrics = 512
            overlap_metrics = 0.5
            n_eval_frames = 4 #There are 4 1-second frames per batch.

            # reset rng state so that each example gets the same state
            torch.random.set_rng_state(rng_state_torch)
            with torch.no_grad():
                # use model
                if is_u_net: 
                    n_hop = int(trained_model.n_fft - trained_model.overlap * trained_model.n_fft)
                    estimated_sources = utils.masking_unets_softmasks(trained_model, mix, f0_hz, n_sources,
                                                                    trained_model.n_fft, n_hop)
                    # estimated_sources = utils.masking_unets2(trained_model, mix, f0_hz, n_sources,
                    #                                          trained_model.n_fft, n_hop)
                    source_estimates = torch.tensor(estimated_sources, device=device, dtype=torch.float32).unsqueeze(0)  # [batch_size, n_source, n_samples]
                    source_estimates_masking = source_estimates.reshape((batch_size * n_sources, n_samples))
                else:
                    mix_estimate, source_estimates = trained_model(mix, f0_hz)
                    # [batch_size * n_sources, n_samples]
                    source_estimates_masking = utils.masking_from_synth_signals_torch(mix, source_estimates, n_fft=2048, n_hop=256)

                # pre-process for post-process calculations
                target_sources = target_sources.transpose(1, 2)  # [batch_size, n_sources, n_samples]
                target_sources = target_sources.reshape((batch_size * n_sources, n_samples))
                source_estimates = source_estimates.reshape((batch_size * n_sources, n_samples))

                # compute SI-SDR masking
                si_sdr_masking = em.si_sdr(target_sources, source_estimates_masking)
                si_sdr_masking = si_sdr_masking.reshape((batch_size, n_sources, -1)).cpu().numpy()

            # Prepare batch_results
            mix_names = [name for _ in range(n_sources * n_eval_frames)]
            voice = [v for v in voices for _ in range(n_eval_frames)]
            eval_frame = [f for _ in range(n_sources * batch_size) for f in range(n_eval_frames)]
            seed_results = [seed] * len(eval_frame)
            si_sdr_results_masking = [si_sdr_masking[b, s, f] for b in range(batch_size) for s in range(n_sources) for f in range(n_eval_frames)]

            batch_results_masking = pd.DataFrame({'mix_name': mix_names, 'eval_seed': seed_results, 'voice': voice, 'eval_frame': eval_frame, 'SI-SDR': si_sdr_results_masking })
            
            eval_results_masking = eval_results_masking.append(batch_results_masking, ignore_index=True)
    return(eval_results_masking)

def print_stats(tag,eval_results_masking):
    means_masking = eval_results_masking.mean(axis=0, skipna=True, numeric_only=True)
    medians_masking = eval_results_masking.median(axis=0, skipna=True, numeric_only=True)
    stds_masking = eval_results_masking.std(axis=0, skipna=True, numeric_only=True)

    print(tag + '_masking')
    print('SI-SDR:', 'mean', means_masking['SI-SDR'], 'median', medians_masking['SI-SDR'], 'std', stds_masking['SI-SDR'])

def add_results_to_log(eval_results_masking,tag,device,test_name,test_index):
    '''
    update(or init) a big csv log.
    columns are : model(tag), device, test_name(baseline, transposition, octave, n_strict_error_tests),
        test_index(also used as test_parameters), SI-SDR mean, SI-SDR median, SI-SDR std
    '''
    try:
        log_df=pd.read_csv('./robustness_tests/robustness_eval_log.csv')
    except(FileNotFoundError):
        log_df=pd.DataFrame(columns=["model","device","test_name","test_index", "SI-SDR_mean", "SI-SDR_median", "SI-SDR_std"])
    new_row = {"model" : tag,
        "device" : device,
        "test_name" : test_name,
        "test_index" : test_index,
        "SI-SDR_mean" : eval_results_masking.mean(axis=0, skipna=True, numeric_only=True)['SI-SDR'],
        "SI-SDR_median" : eval_results_masking.median(axis=0, skipna=True, numeric_only=True)['SI-SDR'],
        "SI-SDR_std": eval_results_masking.std(axis=0, skipna=True, numeric_only=True)['SI-SDR']}
    log_df = log_df.append(pd.Series(new_row), ignore_index=True)
    log_df = log_df[["model","device","test_name","test_index", "SI-SDR_mean", "SI-SDR_median", "SI-SDR_std"]]
    log_df.sort_values(by=["device","model","test_name","test_index"],inplace=True)
    log_df.to_csv('./robustness_tests/robustness_eval_log.csv')

def apply_evaluate_save_test(lambda_func_builder,test_indexes,test_name,path_to_save_results_masking,song_gtf0_dict):
    '''
    Runs needed evals with their test criteria while making sure they haven't been done before and then saving them to a log file
    '''
    for test_index in test_indexes:
        path_test=path_to_save_results_masking +'/{}/{}'.format(test_name,test_index)
        if not os.path.isdir(path_test):
            os.makedirs(path_test, exist_ok=True)
        if len(os.listdir(path_test))==0:  # only run evaluation if directory is empty (evaluation wasn't completed or directory was just created)
            print('\nevaluating :', path_test[len(path_to_save_results_masking):],'\n')
            mess_func=lambda_func_builder(test_index)

            el_rossinyol_gtf0_messed_df=song_gtf0_dict['el_rossinyol'].apply(mess_func)
            locus_iste_gtf0_messed_df=song_gtf0_dict['locus_iste'].apply(mess_func)
            nino_dios_gtf0_messed_df=song_gtf0_dict['nino_dios'].apply(mess_func)
            eval_results_masking=eval_body(n_seeds,test_set,el_rossinyol_gtf0_messed_df,locus_iste_gtf0_messed_df,nino_dios_gtf0_messed_df)

            eval_results_masking.to_pickle(path_test +'/all_results.pandas')

            add_results_to_log(eval_results_masking,tag,device,test_name,test_index)


########################## TESTS ##########################

# Number of seeds
if is_u_net: n_seeds = 1
else: n_seeds = 5
if force_only_one_seed : n_seeds=1

song_gtf0_dict = load_groundTruthF0_files()

'''Baseline test'''
test_name='baseline_gtf0'
if test_name in tests_to_compute:
    def lambda_identity_builder(any):
        return(lambda x : x)
    test_indexes=[0]
    apply_evaluate_save_test(lambda_identity_builder,test_indexes,test_name,path_to_save_results_masking,song_gtf0_dict)

'''Transposition tests'''
test_name='gtf0_transposed'
if test_name in tests_to_compute:
    def lambda_transpose_builder(value):
        '''leaves 0.0 as 0.0 and transposes the rest by value.'''
        return(lambda x : (x!=0)*x+value)
    test_indexes=list(range(-60,60,3))
    apply_evaluate_save_test(lambda_transpose_builder,test_indexes,test_name,path_to_save_results_masking,song_gtf0_dict)

'''Octave tests'''
test_name='gtf0_octaved'
if test_name in tests_to_compute:
    def lambda_octave_builder(value):
        '''leaves 0.0 as 0.0 and octaves the rest by value.'''
        return(lambda x : (x!=0)*x*(2**value))
    test_indexes=list(range(-3,4,1))
    test_indexes.remove(0)
    apply_evaluate_save_test(lambda_octave_builder,test_indexes,test_name,path_to_save_results_masking,song_gtf0_dict)

''' Wrong amount of f0 tracks tests''' #FULL F0 COLUMN ERRORS.
test_name='gtf0_voices_missing'
if test_name in tests_to_compute:
    test_indexes=['s','a','t','b','sa','st','sb','at','ab','tb','sat','sab','stb','atb','satb']
    for test_index in test_indexes:
        path_test=path_to_save_results_masking +'/{}/{}'.format(test_name,str(len(test_index))+test_index)
        if not os.path.isdir(path_test):
            os.makedirs(path_test, exist_ok=True)
        if len(os.listdir(path_test))==0:
            print('\nevaluating :', path_test[len(path_to_save_results_masking):],'\n')
            
            el_rossinyol_gtf0_messed_df=song_gtf0_dict['el_rossinyol'].copy()
            locus_iste_gtf0_messed_df=song_gtf0_dict['locus_iste'].copy()
            nino_dios_gtf0_messed_df=song_gtf0_dict['nino_dios'].copy()

            for v in list(test_index):
                el_rossinyol_gtf0_messed_df[v]=el_rossinyol_gtf0_messed_df[v].apply(lambda x : 0)
                locus_iste_gtf0_messed_df[v]=locus_iste_gtf0_messed_df[v].apply(lambda x : 0)
                nino_dios_gtf0_messed_df[v]=nino_dios_gtf0_messed_df[v].apply(lambda x : 0)

            eval_results_masking=eval_body(n_seeds,test_set,el_rossinyol_gtf0_messed_df,locus_iste_gtf0_messed_df,nino_dios_gtf0_messed_df)

            eval_results_masking.to_pickle(path_test +'/all_results.pandas')

            test_index_for_sort=len(test_index)
            if 's' in test_index:
                test_index_for_sort+=0.0001
            if 'a' in test_index:
                test_index_for_sort+=0.0010
            if 't' in test_index:
                test_index_for_sort+=0.0100
            if 'b' in test_index:
                test_index_for_sort+=0.1000

            add_results_to_log(eval_results_masking,tag,device,test_name,test_index_for_sort)

'''n_strict_error tests''' #FULL F0 ROW ERRORS. No single voice errors.
test_name='gtf0_strict_error_percent'
if test_name in tests_to_compute:
    test_indexes=list(range(99,0,-3))

    random.setstate(random_state) # The random_int sequence needs to be created identical each time.
    randintsequence=[[random.randint(1,100) for _ in range(len(song_gtf0_dict[song].index)) ] for song in ['el_rossinyol','locus_iste','nino_dios'] ]

    for test_index in test_indexes:
        path_test=path_to_save_results_masking +'/{}/{}'.format(test_name,test_index)
        if not os.path.isdir(path_test):
            os.makedirs(path_test, exist_ok=True)
        if len(os.listdir(path_test))==0:
            print('\nevaluating :', path_test[len(path_to_save_results_masking):],'\n')
            
            mask=[[randint>test_index for randint in randintsequence_song ] for randintsequence_song in randintsequence ]

            el_rossinyol_gtf0_messed_df=song_gtf0_dict['el_rossinyol'].multiply(mask[0], axis='index')
            locus_iste_gtf0_messed_df=song_gtf0_dict['locus_iste'].multiply(mask[1], axis='index')
            nino_dios_gtf0_messed_df=song_gtf0_dict['nino_dios'].multiply(mask[2], axis='index')

            eval_results_masking=eval_body(n_seeds,test_set,el_rossinyol_gtf0_messed_df,locus_iste_gtf0_messed_df,nino_dios_gtf0_messed_df)

            eval_results_masking.to_pickle(path_test +'/all_results.pandas')

            add_results_to_log(eval_results_masking,tag,device,test_name,test_index)
