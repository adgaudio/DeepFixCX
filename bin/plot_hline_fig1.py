import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


TT_base = 1851
TT_ours = 99
ACC_base = .82
ACC_base_std=.02
df = pd.DataFrame([
    {'model': 'Rhline+Heart (Ours)',
     'ROC AUC': .79,
     'Training Time': TT_ours,
     'IMR': .17,
     'ODR': '//',
     'color': plt.cm.Set1(1),
     },
    {'model': 'DenseNet121 (baseline)',
     'ROC AUC': ACC_base,
     'Training Time': TT_base,
     'IMR': 1,
     'ODR': 1,
     'color': 'Gray',
     },
])
# Acc vs Speed
fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=200)
ax.set_ylim(.5, 1.)
ax.text(.5, .45, r"$\longleftarrow$ "+f'{TT_base / TT_ours:.02g}x Faster'+r"$\longrightarrow$",
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes, fontsize=24)
ax.hlines(y=ACC_base, xmin=TT_ours, xmax=TT_base, linestyles='solid', colors='gray', linewidth=1, alpha=1)
ax.hlines(y=[ACC_base-ACC_base_std, ACC_base+ACC_base_std], xmin=TT_ours, xmax=TT_base, linestyles='dashed', colors='gray', linewidth=.5, alpha=.6)
ax.hlines(y=[ACC_base-ACC_base_std*2, ACC_base+ACC_base_std*2], xmin=TT_ours, xmax=TT_base, linestyles='dotted', colors='gray', linewidth=.5, alpha=.2)
df.set_index('model').plot.scatter('Training Time', 'ROC AUC', ax=ax, s=84, c='color')
ax.legend(['Baseline DenseNet121', r'$\pm 1$ std', r'$\pm 2$ std', 'RHLine+Heart (Ours)', 'z'], loc='lower right')
fig.tight_layout()
fig.savefig('hline_acc_vs_speed.png', bbox_inches='tight')


# IMR vs ODR
# see plot_hline_archdiagramfigs

