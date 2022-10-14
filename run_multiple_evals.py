'''run evaluations on multiple models'''

import os

tags = [
        'unsupervised_2s_satb_bcbq_mf0_1',
        'unsupervised_4s_satb_bcbq_mf0_1_again'
        ]

# eval_mode='default' # default evaluation
eval_mode='fast' # default evaluation

if eval_mode=='default':
    for tag in tags:
        print("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute all".format(tag))
        os.system("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute all".format(tag))

elif eval_mode=='fast':
    for tag in tags:
        print("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute SI-SDR_mask".format(tag))
        os.system("python eval.py --tag '{}' --f0-from-mix --test-set 'CSD' --show-progress --compute SI-SDR_mask".format(tag))