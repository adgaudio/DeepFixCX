#!/usr/bin/env python
# plot_heatmap_for_anonymity_score.py

import torch as T
import pandas as pd
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import re


def parse_filename(fp):
    return re.search(
        r'(?P<n_bootstrap>\d+)-(?P<n_patients>\d+)-(?P<wavelet>.*?)-(?P<Level>\d+)-(?P<Patch_Size>\d+)-(?P<Patch_Features>\S+).pth',
        fp).groupdict()


if __name__ == "__main__":
    n_patients = int(sys.argv[1])

    fps = glob.glob(f'results/anonymity_scores/6-{n_patients}-db1-*-l1.pth')
    _dfdata = []
    print(fps)
    ks_tests = {}
    for fp in fps:
        print(fp)
        f_name = os.path.basename(fp)
        dct = parse_filename(fp)
        args = f_name.split('-')
        data = T.load(fp, map_location='cpu')
        test_statistic = np.median([x.statistic for x in data['ks_tests']])
        ks_tests[fp] = data['ks_tests']
        #  ks_stats = test_statistic[0]
        # print(test_statistic[1] < 1e-4)
        #  print(ks_stats)
        _dfdata.append({
            'test_statistic': test_statistic,
            **{k: int(v) if v.isdigit() else v for k,v in dct.items()}})

    df = pd.DataFrame(_dfdata)
    df['Patch Size, P'] = df['Patch_Size']
    df['Wavelet Level, J'] = df['Level']

    pivot_table = pd.pivot_table(
        df, values='test_statistic',
        index='Patch Size, P', columns='Wavelet Level, J')
    fig, ax = plt.subplots()
    sns.heatmap(
        data=pivot_table, annot=True, norm=plt.cm.colors.PowerNorm(2),
        cmap='YlGnBu_r', linewidths=.5, ax=ax, cbar=False)
    #  fig.suptitle('Privacy: Re-identification Score')
    ax.set_title('Privacy: Re-identification Score')
        #  .set_title('ks statistics')
    save_fp = f'results/plots/heatmap_anonymity_ks_{n_patients}.png'
    fig.savefig(save_fp, bbox_inches='tight')
    pivot_table.to_csv(save_fp.replace('.png', '.csv'))
    print(f'saved to:  {save_fp}')
    #  plt.show()
