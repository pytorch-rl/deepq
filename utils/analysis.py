import pandas as pd
import seaborn as sns
import os
import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--trials_dir_path',
                        required=True)

    return parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    args = parse_args()
    progresses = []

    for trial_dir in os.listdir(args.trials_dir_path):
        if trial_dir.startswith('trial'):
            log_file_path = '/'.join([args.trials_dir_path, trial_dir, 'progress.txt'])
            cur_df = pd.read_csv(log_file_path, sep="\t")
            cur_trial_num = int(trial_dir.split('_')[-1])
            cur_df['trial'] = pd.Series(np.ones_like(cur_df.Epoch.values) * cur_trial_num)
            progresses.append(cur_df)

    progresses_df = pd.concat(progresses)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['figure.titleweight'] = 'bold'

    sns.set()
    # sns.catplot(x='Epoch',
    #              y='EpisodeDuration',
    #              data=progresses_df)
    ax = sns.lineplot(x='Epoch',
                 y='MeanEpisodeDuration',
                 hue='trial',
                 data=progresses_df,
                 estimator=None)
    # giving labels to x-axis and y-axis
    # giving labels to x-axis and y-axis
    ax.set(xlabel='Episodes trained', ylabel='Mean Episode Duration')
    # giving title to the plot
    # giving title to the plot
    font = {'family': 'serif',
            'weight': 'bold',
            'size'  : 16,
            }
    plt.title('Mean Episode Duration Vs Episodes Trained', fontdict=font)
    # estimator=None)
    plt.show()
    print('')

# if __name__ == '__main__':
#
#     args = parse_args()
#     progresses = []
#
#     for trial_dir in os.listdir(args.trials_dir_path):
#         if trial_dir.startswith('trial'):
#             log_file_path = '/'.join([args.trials_dir_path, trial_dir, 'progress.txt'])
#             progresses.append(pd.read_csv(log_file_path, sep = "\t"))
#
#     progresses_df = pd.concat(progresses)
#     sns.set()
#     sns.lineplot(x=progresses_df.Epoch, y=progresses_df.EpisodeDuration, ci='sd', estimator='mean')
#     # sns.lineplot(x=progresses_df.Epoch, y=progresses_df.MeanEpisodeDuration, ci='sd', estimator='mean')
#     print('')