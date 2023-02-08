"""Script for performing the analysis of the paper."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

mode = 'statistics'
# Paths to the different log files.
# --- 3 var
data_paths = ['experiments/3lin_intrew_combined/eval_MCD.npy',
              'experiments/3lin_intrew_combined/eval_MCD_obs.npy',
              'experiments/3lin_intrew_combined/eval_dcdi_10000_3333_50scms.npy',
              'experiments/3lin_intrew_combined/eval_enco_10000_3333_50ep.npy',
              'experiments/3lin_intrew_combined/eval_notears_10000obs_50scms.npy',
              'experiments/3lin_intrew_combined/eval_random.npy']
# --- 3var intervention design
data_paths = ['experiments/3lin_intrew_combined/eval_MCD.npy',
              'experiments/3lin_intrew_combined/eval_dcdi_5_17_50scms.npy',
              'experiments/3lin_intrew_combined/eval_enco_4_17_50ep_50scms.npy',
              'experiments/3lin_intrew_combined/eval_notears_20samples_50scms.npy']

#----- 4 var
# data_paths = ['experiments/4lin_intrew_combined/eval_MCD.npy',
#               'experiments/4lin_intrew_combined/eval_dcdi_10000_3333_50scms.npy',
#               'experiments/4lin_intrew_combined/enco_4lin_1000_333_30_70scm_eval.npy',
#               'experiments/4lin_intrew_combined/eval_notears_10000_50scms.npy',
#               'experiments/4lin_intrew_combined/eval_random.npy']
data_labels = ['MCD (ours)',
               #'MCD-O (ours)',
               'DCDI',
               #'DCDI few',
               'ENCO',
               #'ENCO few',
               'NOTEARS',
               #'NOTEARS few',
               #'random',
                ]

# Plot the reward over time
if mode == 'learning_curve':
    data = {}
    data['MCD'] = np.load('experiments/3lin_intrew_combined/log.npz')
    data['MCD-O'] = np.load('experiments/3lin_intrew_obs/log.npz')
    sns.lineplot(data=data['MCD'], y='results_val', x="timesteps")
    sns.lineplot(data=data['MCD-O'], y='results_val', x="timesteps")
    plt.show()

# Create boxplots of the dSHDs
elif mode == 'boxplots':
    shds = {}
    times = {}
    for i in range(len(data_paths)):
        data = np.load(data_paths[i])
        shds[data_labels[i]] = [d[0] for d in data]

    sns.boxplot(data=[s[:50] for s in shds.values()],
                showmeans=True,
                meanprops={"marker": "x",
                           "markeredgecolor": "black",
                           "markersize": "8"}).set(xticklabels=data_labels, ylabel='dSHD',
                                                   title="3 Endogenous Variables", ylim=[0, 9])
    plt.savefig('experiments/3lin_intrew_combined/delme.pdf')
    plt.show()

# Perform significance tests on the SHDs and runtimes
elif mode == 'statistics':
    times = {}
    shds = {}
    n_scms = 50
    for i in range(len(data_paths)):
        data = np.load(data_paths[i])
        shds[data_labels[i]] = [d[0] for d in data]
        times[data_labels[i]] = [d[1] for d in data]

    print('times')
    for key, value in times.items():
        value = np.array(value[:n_scms])
        print(key, value.mean(), np.median(value), value.std())

    print('shds')
    for key, value in shds.items():
        value = np.array(value[:n_scms])
        print(key, value.mean(), np.median(value), value.std())

    for label in data_labels:
        if label != 'MCD (ours)':
            print(label, wilcoxon(shds['MCD (ours)'][:n_scms], shds[label][:n_scms],
                                  alternative='less',
                                  zero_method='pratt'))