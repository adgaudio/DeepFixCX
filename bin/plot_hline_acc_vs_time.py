import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch as T

from heartspot.plotting import arrow_with_text_in_middle

plt.rcParams.update({"text.usetex": True,})


# IMR vs ODR
# see plot_hline_archdiagramfigs
def clampq(x: T.Tensor, quantiles=(0,.99)):
    if isinstance(quantiles, (list, tuple)):
        quantiles = x.new_tensor(quantiles)
    return x.clamp(*x.quantile(quantiles))



timecol = 'Training Throughput (Sec/15k Imgs)'
TT_base = 143  # 15k/sec
TT_base_std = 4
TT_ours = 16
#  timecol = 'Training Time'
#  TT_base = 20521.105421411226
#  TT_ours = 2824.6117766582643
ACC_base = .829  # TODO: update this
ACC_base_std=.017  # TODO: update this
# TODO: the numbers here.
# note: the order in which values appear affects the shade of their color.
df = pd.DataFrame.from_records({
    #  'ODR': df['On-Disk Compression Ratio'],
    #  'IMR': df['In-Memory Compression Ratio'],
    'ROC AUC': pd.Series({
        'Median+(RH)Line+Heart (MLP)': 0.771,
        'HLine (DenseNet)': .842,
        'Median+HLine (DenseNet)': .837,
        'Median+(RH)Line+Heart (DenseNet)': 0.816,
        '(RH)Line+Heart (MLP)': 0.781,
        'RLine (MLP)': 0.772,
        'HLine (MLP)': 0.722,
        'Heart (MLP)': 0.722,
        '(RH)Line+Heart (DenseNet)': 0.819,
        'RLine (DenseNet)': 0.794,
        'Heart (DenseNet)': 0.829,
    }),
    'Training Throughput (Sec/15k Imgs)': pd.Series({
        'Median+(RH)Line+Heart (MLP)': 94,
        'HLine (DenseNet)': 155,
        'Median+HLine (DenseNet)': 136,
        'Median+(RH)Line+Heart (DenseNet)': 128,
        '(RH)Line+Heart (MLP)': 94,
        'RLine (MLP)': 20,
        'HLine (MLP)': 16,
        'Heart (MLP)': 22,
        '(RH)Line+Heart (DenseNet)': 154,
        'RLine (DenseNet)': 158,
        'Heart (DenseNet)': 159,
        }),
    'Training Time': pd.Series({
        'Median+(RH)Line+Heart (MLP)': 20843.88001555479,
        'HLine (DenseNet)': 21542.090708569816,
        'Median+HLine (DenseNet)': 14389.657654,
        'Median+(RH)Line+Heart (DenseNet)': 13468.848903872273,
        '(RH)Line+Heart (MLP)': 2824.6117766582643,
        'RLine (MLP)': 3289.5340505003714,
        'HLine (MLP)': 3188.83208707591,
        'Heart (MLP)': 3188.757367484832,
        '(RH)Line+Heart (DenseNet)': 20546.70023594248,
        'RLine (DenseNet)': 20003.923889087047,
        'Heart (DenseNet)': 19128.628174996607,
        }),
    'color': pd.Series({
        'Median+(RH)Line+Heart (MLP)': 'black',
        'HLine (DenseNet)': plt.cm.Set1(4),
        'HLine (DenseNet)': plt.cm.Set1(9),
        'Median+(RH)Line+Heart (DenseNet)': plt.cm.Set1(8),
        '(RH)Line+Heart (MLP)': plt.cm.Set2(3),
        'RLine (MLP)': plt.cm.Set2(1),
        'HLine (MLP)': plt.cm.Set2(0),
        'Heart (MLP)': plt.cm.Set2(2),
        '(RH)Line+Heart (DenseNet)': plt.cm.Set1(7),
        'RLine (DenseNet)': plt.cm.Set1(5),
        'Heart (DenseNet)': plt.cm.Set1(6),
    }),
})
df = pd.concat([df, pd.DataFrame({
     'ROC AUC': ACC_base,
     'Training Throughput (Sec/15k Imgs)': TT_base,
     'IMR': 1,
     'ODR': 1,
     'color': 'Gray',
     }, index=pd.Index(['DenseNet121 (baseline)'], name='Model'))])
df.index.name = "Model"

# TODO get perf from the cdfs:
# run in terminal:
#  $ simplepytorch_plot "5.HL8"  -c ROC_AUC  --mode1-subplots
# then get the data:
#  z = cdfs.query('epoch <=300')
#  y = z.loc[z.groupby('run_id')['val_ROC_AUC Cardiomegaly'].idxmax()]['test_ROC_AUC Cardiomegaly'].sort_values()
#  sec = cdfs.groupby('run_id')['seconds_training_epoch'].mean()
#  timings = (y.reset_index('epoch')['epoch'] * sec).sort_values().rename('timing')
#  x = pd.concat([y.reset_index('epoch', drop=True), timings], axis=1)
#  x.index = x.index\
        #  .str.replace('\+densenet.', lambda x: ' (DenseNet)')\
        #  .str.replace('5.HL8.', lambda x: '')\
        #  .str.replace("heart", lambda x: 'Heart')\
        #  .str.replace('rhline', lambda x: '(RH)Line')\
        #  .str.replace('rline', lambda x: 'RLine')\
        #  .str.replace('hline', lambda x: 'HLine')\
        #  .str.replace('median', lambda x: 'Median')\
        #  .str.rstrip('.')


# Acc vs Speed
fig, ax = plt.subplots(1,1, figsize=(4,2.5), dpi=200)
ax.set_ylim(.5, 1.)
#  ax.text(.55, .85, r"$\longleftarrow$ "+f'{TT_base / TT_ours:.02g}x Faster'+r" $\rightarrow$",
        #  horizontalalignment='center', verticalalignment='center',
        #  transform=ax.transAxes, fontsize=26)

arrow_with_text_in_middle(
    f'{TT_base / TT_ours:.01g}$x$ Faster',
    left_xy=(TT_ours,.90), text_xy=((TT_base+TT_ours)/2, .90), right_xy=(TT_base,.90),
    arrowprops={'lw': 2}, fontsize=24, ax=ax
)


#  df.set_index('model').plot.scatter(timecol, 'ROC AUC', ax=ax, s=84, c='color')
ax.hlines(y=ACC_base, xmin=0, xmax=df[timecol].max(), linestyles='solid', colors='gray', linewidth=1, alpha=1)
sns.scatterplot(x=timecol, y='ROC AUC', hue='Model', data=df.reset_index().loc[df.index.str.contains('MLP')], s=24*4, ax=ax, legend=None, palette='Blues_r')
sns.scatterplot(x=timecol, y='ROC AUC', hue='Model', data=df.reset_index().loc[df.index.str.contains('DenseNet')], s=24*4, ax=ax, legend=None, palette='Greens_r')
ax.legend(['Baseline DenseNet121 $\pm$ 2 std', r'\textit{HeartSpot} with MLP', r'\textit{HeartSpot} with DenseNet121'], loc='lower center', ncol=1)
#  ax.scatter(TT_base, ACC_base, c='Gray', s=60*2, label='Baseline DenseNet121')
ax.vlines(TT_base, .5, ACC_base, colors='gray', linewidth=1, label='Baseline DenseNet121')
ax.vlines([TT_base-TT_base_std*2, TT_base+TT_base_std*2], .5, ACC_base, colors='gray', linewidth=.5, linestyles='dashed', alpha=.6, label='Baseline DenseNet121')
#  ax.hlines(y=[ACC_base-ACC_base_std, ACC_base+ACC_base_std], xmin=0, xmax=TT_base, linestyles='dashed', colors='gray', linewidth=.5, alpha=.6)
ax.hlines(y=[ACC_base-ACC_base_std*2, ACC_base+ACC_base_std*2], xmin=0, xmax=df[timecol].max(), linestyles='dashed', colors='gray', linewidth=.5, alpha=.6)
#  fig.tight_layout()
fig.savefig('hline_acc_vs_speed.png', bbox_inches='tight')
