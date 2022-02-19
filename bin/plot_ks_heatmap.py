# plot_heatmap_for_anonymity_score.py

import torch as T
import pandas as pd
import seaborn as sns
import glob
import os
import matplotlib.pyplot as plt


fps = glob.glob('results/anonymity_scores/*_coif1-*-*-l1.pth')
data = []
print(fps)
for fp in fps:
    f_name = os.path.basename(fp)
    args = f_name.split('-')
    level = int(args[1])
    patch_size = int(args[2])
    test_statistic = T.load(fp)['ks_test']
    ks_stats = test_statistic[0]
    print(test_statistic[1] < 1e-4)
    data.append({'level': level, 'patchsize': patch_size, 'test_statistic': ks_stats})

df = pd.DataFrame(data)

pivot_table = pd.pivot_table(df, values='test_statistic',index='patchsize',columns='level')
ax = sns.heatmap(data=pivot_table,annot=True, cmap='RdYlGn',linewidths=.5)
ax.set_title('ks statistics')
plt.show()


