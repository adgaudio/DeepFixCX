import pandas as pd
from deepfix import plotting as P
import matplotlib.pyplot as plt

stats = [
    #
    # CheXpert
    #
    {
        'Dataset': 'Chest X-ray (CheXpert)', 'Model': 'Baseline CNN',
        'Accuracy (ROC AUC)': .868, 'Accuracy (errorbar)': 0.006,
        'In-Memory Compression Ratio': 100,
        'Privacy (SSIM)': 1,
        'Train Speed (sec)': 645,
        'optimized_for': 'baseline',
    },
    {
        'Dataset': 'Chest X-ray (CheXpert)', 'Model': 'DeepFixCX+CNN',
        'Accuracy (ROC AUC)': .867, 'Accuracy (errorbar)': 0.006,
        'In-Memory Compression Ratio': 24,
        'Privacy (SSIM)': .717,
        'Train Speed (sec)': 217,
        'optimized_for': 'accuracy',
        'J': 1, 'P': 79,
    },
    {
        'Dataset': 'Chest X-ray (CheXpert)', 'Model': 'DeepFixCX+CNN',
        'Accuracy (ROC AUC)': .855, 'Accuracy (errorbar)': None,
        'In-Memory Compression Ratio': 1.410,
        'Privacy (SSIM)': .670,
        'optimized_for': 'compression',
        'J': 1, 'P': 19,
    },
    #
    # KimEye
    #
    {
        'Dataset': 'Glaucoma (Kimeye)', 'Model': 'Baseline CNN',
        'Accuracy (ROC AUC)': .924, 'Accuracy (errorbar)': 0.026,
        'In-Memory Compression Ratio': 100,
        'Privacy (SSIM)': 1,
        'Train Speed (sec)': 11,
        'optimized_for': 'baseline',
    },
    {
        'Dataset': 'Glaucoma (Kimeye)', 'Model': 'DeepFixCX+CNN',
        'Accuracy (ROC AUC)': .947, 'Accuracy (errorbar)': 0.015,
        'In-Memory Compression Ratio': 11.1,
        'Privacy (SSIM)': .971,
        'Train Speed (sec)': 6,
        'J': 4, 'P': 5,
        'optimized_for': 'accuracy',
    },
    {
        'Dataset': 'Glaucoma (Kimeye)', 'Model': 'DeepFixCX+CNN',
        'Accuracy (ROC AUC)': .942, 'Accuracy (errorbar)': None,
        'In-Memory Compression Ratio': 0.694,
        'Privacy (SSIM)': 0.975,
        'optimized_for': 'compression',
        'J': 2, 'P': 5,
    },
    #
    # IntelMobileODT
    #
    {
        'Dataset': 'Cervix (IntelMobileODT)', 'Model': 'Baseline CNN',
        'Accuracy (ROC AUC)': 0.741, 'Accuracy (errorbar)': 0.014,
        'In-Memory Compression Ratio': 100,
        'Privacy (SSIM)': 1,
        'Train Speed (sec)': 37,
        'optimized_for': 'baseline',
    },
    {
        'Dataset': 'Cervix (IntelMobileODT)', 'Model': 'DeepFixCX+CNN',
        'Accuracy (ROC AUC)': 0.763, 'Accuracy (errorbar)': 0.009,
        'In-Memory Compression Ratio': 85,
        'Privacy (SSIM)': .558,
        'Train Speed (sec)': 36,
        'optimized_for': 'accuracy',
        'J': 5, 'P': 5,
    },
    {
        'Dataset': 'Cervix (IntelMobileODT)', 'Model': 'DeepFixCX+CNN',
        'Accuracy (ROC AUC)': 0.740, 'Accuracy (errorbar)': None,
        'In-Memory Compression Ratio': 0.653,
        'Privacy (SSIM)': .807,
        'Train Speed (sec)': None,
        'optimized_for': 'compression',
        'J': 1, 'P': 7,
    },
]

df = pd.DataFrame(stats)

# split out the legend into two parts.  by color. by style.

colors = []
fig, ax = plt.subplots(figsize=(6.4,4.0))
for n, (dataset, tmp) in enumerate(df.groupby('Dataset')):
    arrow_start_end = []
    for (model, _), vals in tmp.groupby(['Model', 'optimized_for'], sort=False):
        marker = 'o' if 'DeepFixCX' in model else 'x'
        label = f'{model} // {dataset}'
        color = plt.cm.Set1(n)
        x = vals['In-Memory Compression Ratio'].item()
        y = vals['Accuracy (ROC AUC)'].item()
        ax.scatter(x, y, marker=marker, label=label, color=color)
        # plt.errorbar(x, y, yerr=vals['Accuracy (errorbar)'], ecolor='gray',
        #              elinewidth=1, capsize=10 )
        arrow_start_end.append((vals['In-Memory Compression Ratio'].item(), vals['Accuracy (ROC AUC)'].item()))
        colors.append((dataset, color))
    print(arrow_start_end)
    for arrow_left_xy in arrow_start_end[1:]:
        # P.arrow_with_text_in_middle(
        #     None, *arrow_start_end, arrowstyle=('->', '-'))
        ax.annotate(
            None, arrow_left_xy, arrow_start_end[0],
            arrowprops={'arrowstyle': '->', 'color': 'lightgray'},
            horizontalalignment='center', verticalalignment='center',
            fontsize='xx-large')
    # baseline
    ax.hlines(arrow_start_end[0][1], -1, 105, linewidth=1, linestyle='--', color='lightgrey', label=None)
# legend: color for each dataset
ax.add_artist(ax.legend(
    title='Dataset',
    handles=[plt.Line2D([], [], linewidth=0, marker='o', color=color, label=dataset, markersize=10) for dataset, color in set(colors)], loc='lower right'))
# legend: style for deepfix / no deepfix
ax.legend(handles=[
    plt.Line2D([], [], color='black', linewidth=0, marker='o', markersize=10, label='with DeepFixCX'),
    plt.Line2D([], [], color='black', linewidth=0, marker='x', markersize=10, label='without DeepFixCX'),], loc='lower left')
ax.set_xlabel('In-Memory Compression Ratio')
ax.set_ylabel('Accuracy (ROC AUC)')
ax.set_xlim(-5,125)
ax.set_ylim(.55,1.05)

P.arrow_with_text_in_middle(
    '153x Smaller Images', (0.653, 1.01), (100, 1.01), arrowstyle=('->', '-'))
P.arrow_with_text_in_middle(
    None, None, (113, .95), (113, .741), arrowstyle=('->', '->'))
ax.text(x=115, y=.85, s='More Accurate',
        fontsize='xx-large', rotation=-90,verticalalignment='center',)
plt.show()

fig.savefig('./results/plots/deepfix_fig1_acc_vs_imr.png', bbox_inches='tight', dpi=300)


list(df.groupby('Dataset'))
