#!/usr/bin/env python
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import sys


experiment_name = sys.argv[1]  # '2.C21'
df = pd.read_csv(f'./results/plots/roc_auc/test_{experiment_name}.csv')

regex = r'J=(?P<J>\d+).P=(?P<P>\d+)'
model_hyperparams = df["run_id"].str.extractall(regex).rename(columns={'J': 'Wavelet Level, J', 'P': 'Patch Size, P'}).astype('int')
assert model_hyperparams.shape[0] > 0, 'sanity check: the experiments ids should contain a string like "waveletmlp:1:14:4:3:1:2"'
df = df.join(model_hyperparams.reset_index('match', drop=True), how='outer')


# generate the plot
#  note:  pivot table implicitly computes the mean when the sanity check fails
for col, filename_substring in [('ROC AUC LEADERBOARD', 'rocauc'),
                                ('BAcc LEADERBOARD', 'bacc_leaderboard')]:
    heatmap_data = df.pivot_table(col, "Patch Size, P", "Wavelet Level, J")
    fig, ax = plt.subplots(figsize=(6,2.5), dpi=300)
    ax = sns.heatmap(data=heatmap_data, cmap='RdYlGn', annot=True, fmt='.03f', cbar=False, ax=ax)
    #  ax.set_title(f'Predictive Performance: Test {col}')

    # save the plot to file
    savefp = f'./results/plots/heatmap_perf_{filename_substring}__{experiment_name}.png'
    if os.path.exists(savefp):
        os.remove(savefp)
    ax.figure.savefig(savefp, bbox_inches='tight')
    heatmap_data.to_csv(savefp.replace('.png', '.csv'))
    print(f'save to: {savefp}')
