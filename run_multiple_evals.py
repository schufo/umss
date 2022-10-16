'''run evaluations on multiple models'''

import os

models_to_evaluate = [
        'unsupervised_2s_satb_bcbq_mf0_1',
        'unsupervised_4s_satb_bcbq_mf0_1_again'
        ]

# eval_mode='default' # default evaluation
eval_mode='fast' # fast evaluation

if eval_mode=='original':
    for tag in models_to_evaluate:
        print("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD'".format(tag))
        os.system("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD'".format(tag))

elif eval_mode=='default':
    for tag in models_to_evaluate:
        print("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute all".format(tag))
        os.system("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute all".format(tag))

elif eval_mode=='fast':
    for tag in models_to_evaluate:
        print("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute SI-SDR_mask".format(tag))
        os.system("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute SI-SDR_mask".format(tag))