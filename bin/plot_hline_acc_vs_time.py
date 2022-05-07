import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch as T

plt.rcParams.update({"text.usetex": True,})


# IMR vs ODR
# see plot_hline_archdiagramfigs
def clampq(x: T.Tensor, quantiles=(0,.99)):
    if isinstance(quantiles, (list, tuple)):
        quantiles = x.new_tensor(quantiles)
    return x.clamp(*x.quantile(quantiles))



TT_base = 158  # TODO
TT_base_std = 1  # TODO
TT_ours = 16  # TODO
ACC_base = .82  # TODO: update this
ACC_base_std=.02
# TODO: the numbers here.
# note: the order in which values appear affects the shade of their color.
df = pd.DataFrame.from_records({
    #  'ODR': df['On-Disk Compression Ratio'],
    #  'IMR': df['In-Memory Compression Ratio'],
    'ROC AUC': pd.Series({
        'Median+(RH)Line+Heart (MLP)': 0.82,  # TODO
        'Median+(RH)Line+Heart (DenseNet)': 0.82,
        '(RH)Line+Heart (MLP)': 0.78,
        'RLine (MLP)': 0.77,
        'HLine (MLP)': 0.72,
        'Heart (MLP)': 0.72,
        '(RH)Line+Heart (DenseNet)': 0.82,
        'RLine (DenseNet)': 0.79,
        'HLine (DenseNet)': 0,  # TODO
        'Heart (DenseNet)': 0.83,
    }),
    'Training Time': pd.Series({  # TODO
        'Median+(RH)Line+Heart (MLP)': 94.44,  # TODO
        'Median+(RH)Line+Heart (DenseNet)': 128,
        '(RH)Line+Heart (MLP)': 94,
        'RLine (MLP)': 20,
        'HLine (MLP)': 16,
        'Heart (MLP)': 22,
        '(RH)Line+Heart (DenseNet)': 154,
        'RLine (DenseNet)': 158,
        'HLine (DenseNet)': 0,  # TODO
        'Heart (DenseNet)': 159,
        }),
    'color': pd.Series({
        'Median+(RH)Line+Heart (MLP)': 'black',
        'Median+(RH)Line+Heart (DenseNet)': plt.cm.Set1(8),
        '(RH)Line+Heart (MLP)': plt.cm.Set2(3),
        'RLine (MLP)': plt.cm.Set2(1),
        'HLine (MLP)': plt.cm.Set2(0),
        'Heart (MLP)': plt.cm.Set2(2),
        '(RH)Line+Heart (DenseNet)': plt.cm.Set1(7),
        'RLine (DenseNet)': plt.cm.Set1(5),
        'HLine (DenseNet)': plt.cm.Set1(4),
        'Heart (DenseNet)': plt.cm.Set1(6),
    }),
})
df = df.append(pd.DataFrame({
     'ROC AUC': ACC_base,
     'Training Time': TT_base,
     'IMR': 1,
     'ODR': 1,
     'color': 'Gray',
     }, index=pd.Index(['DenseNet121 (baseline)'], name='Model'))
)
df.index.name = "Model"

# TODO get perf from the cdfs:
#  y = cdfs.loc[
#      cdfs.groupby('run_id')['val_ROC_AUC Cardiomegaly'].idxmax()
#  ]['test_ROC_AUC Cardiomegaly'].sort_values().reset_index('epoch', drop=True)
#  y2 = cdfs.groupby('run_id')['seconds_training_epoch'].mean()
#  y2
#  y.index = y.index\
#          .str.replace('\+densenet.', lambda x: ' (DenseNet)')\
#          .str.replace('5.HL8.', lambda x: '')\
#          .str.replace("heart", lambda x: 'Heart')\
#          .str.replace('rline', lambda x: 'RLine')\
#          .str.replace('hline', lambda x: 'HLine')\
#          .str.replace('rhline', lambda x: '(RH)Line')\
#          .str.replace('median', lambda x: 'Median')\
#          .str.rstrip('.')


# Acc vs Speed
fig, ax = plt.subplots(1,1, figsize=(4,2.5), dpi=200)
ax.set_ylim(.5, 1.)
#  ax.text(.55, .85, r"$\longleftarrow$ "+f'{TT_base / TT_ours:.02g}x Faster'+r" $\rightarrow$",
        #  horizontalalignment='center', verticalalignment='center',
        #  transform=ax.transAxes, fontsize=26)

def arrow_with_text_in_middle(
        text, left_xy, text_xy, right_xy, arrowprops=None, fontsize='xx-large', **text_kwargs):
    """Draw arrow like   <----  TEXT  ---->  where fontsize controls the size of arrow"""
    text_kwargs = dict(
        horizontalalignment='center', verticalalignment='center', fontsize=fontsize, **text_kwargs)
    _arrowprops = dict(arrowstyle='->')
    _arrowprops.update(arrowprops if arrowprops is not None else {})
    arrowprops = _arrowprops
    if left_xy:
        ax.annotate(
            text, xy=left_xy, xytext=text_xy,
            arrowprops=arrowprops, **text_kwargs)
        alpha = 0
    else:
        alpha = 1
    if right_xy:
        ax.annotate(
            text, xytext=text_xy, xy=right_xy,
            arrowprops=arrowprops, alpha=alpha, **text_kwargs)

arrow_with_text_in_middle(
    f'{TT_base / TT_ours:.02g}$x$ Faster',
    left_xy=(16,.90), text_xy=(90, .90), right_xy=(TT_base,.90),
    arrowprops={'lw': 1}, fontsize=26,
)


#  df.set_index('model').plot.scatter('Training Time', 'ROC AUC', ax=ax, s=84, c='color')
ax.hlines(y=ACC_base, xmin=0, xmax=TT_base, linestyles='solid', colors='gray', linewidth=1, alpha=1)
sns.scatterplot(x='Training Time', y='ROC AUC', hue='Model', data=df.reset_index().loc[df.index.str.contains('MLP')], s=24*4, ax=ax, legend=None, palette='Blues_r')
sns.scatterplot(x='Training Time', y='ROC AUC', hue='Model', data=df.reset_index().loc[df.index.str.contains('DenseNet')], s=24*4, ax=ax, legend=None, palette='Greens_r')
ax.legend(['Baseline DenseNet121 $\pm$ 2 std', r'\textit{HeartSpot} with MLP', r'\textit{HeartSpot} with DenseNet121'], loc='lower center', ncol=1)
#  ax.scatter(TT_base, ACC_base, c='Gray', s=60*2, label='Baseline DenseNet121')
ax.vlines(TT_base, .5, ACC_base, colors='gray', linewidth=1, label='Baseline DenseNet121')
ax.vlines([TT_base-TT_base_std*2, TT_base+TT_base_std*2], .5, ACC_base, colors='gray', linewidth=.5, linestyles='dashed', alpha=.6, label='Baseline DenseNet121')
#  ax.hlines(y=[ACC_base-ACC_base_std, ACC_base+ACC_base_std], xmin=0, xmax=TT_base, linestyles='dashed', colors='gray', linewidth=.5, alpha=.6)
ax.hlines(y=[ACC_base-ACC_base_std*2, ACC_base+ACC_base_std*2], xmin=0, xmax=TT_base, linestyles='dashed', colors='gray', linewidth=.5, alpha=.6)
#  fig.tight_layout()
fig.savefig('hline_acc_vs_speed.png', bbox_inches='tight')
