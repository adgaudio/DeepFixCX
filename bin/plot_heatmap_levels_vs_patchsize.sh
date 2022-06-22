#!/usr/bin/env bash
#
# Generate a heatmap of the wavelet levels vs patch size  for waveletmlp models.
# expects experiment ids to contain substring like 'J=1.P=21'  where 1 and 21
# are positive integers
#
# use like this    $  ./this_script.sh  "experiment_id_regex"
#
# or for chexpert:
#     $  chexpert=true ./this_script.sh  "experiment_id_regex"
#
# Note: it's a cobbled together script.  If there are any errors, they will
# appear in console output.  Watch out for them.  The bash exit code isn't reliable.
#
runid_regex=${1}
shift
baseline_score=${1}
shift
patch_sizes=${*}
shift

python -m simplepytorch.plot_perf $runid_regex --mode 0 <<EOF

# select the subset of data we want to visualize.
col1 = 'val_ROC_AUC AVG'
col2 = 'test_ROC_AUC AVG'

if os.environ.get('chexpert', False) == 'true':
    cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    cols_val = [f'val_ROC_AUC {x}' for x in cols]
    cols_test = [f'test_ROC_AUC {x}' for x in cols]
    cdfs['val_ROC_AUC AVG'] =  cdfs[cols_val].mean(1) # the leaderboard average, not the 14 classes average (because test set is too small)
    cdfs['test_ROC_AUC AVG'] =  cdfs[cols_test].mean(1) # the leaderboard average, not the 14 classes average (because test set is too small)

if 'filename' in cdfs.index.names:
    df = cdfs.loc[cdfs.groupby(['run_id', 'filename'])[col1].idxmax()].groupby('run_id')[col2].mean().sort_values()
    print(df)
else:
    df = cdfs.loc[cdfs.groupby(["run_id"])[col1].idxmax()].sort_values(col2).reset_index()

# extract the model hyper parameters from the experiment id and join to the data we want to visualize.
# ASSUMPTION: the experiment id contains a string like "waveletmlp:300:1:14:4:3:1:2"
# cols = "mlp_channels, in_ch, out_ch, wavelet_levels, patch_size, in_ch_mul, mlp_depth".split(", ")
# regex = "waveletmlp:" + ":".join(f"(?P<{col}>\d+)" for col in cols)

regex = 'J=(?P<J>\d+).P=(?P<P>\d+)'

model_hyperparams = df.reset_index()["run_id"].str.extractall(regex).rename(columns={'J': 'Wavelet Level, J', 'P': 'Patch Size, P'}).astype('int')
assert model_hyperparams.shape[0] > 0, 'sanity check: the experiments ids should contain a string like "waveletmlp:1:14:4:3:1:2"'
df = df.to_frame().reset_index().join(model_hyperparams.reset_index('match', drop=True), how='outer')

patch_sizes="$patch_sizes"
if patch_sizes:
    patch_sizes=list(int(x) for x in patch_sizes.split())
    df = df.loc[df['Patch Size, P'].isin(patch_sizes)]

# you can disable this sanity check if it fails.  Failure would mean you tested
# two models with wavelet_levels=A and patch_size=B for some (A,B)
#  ... at some point, we want to make this fail because we will want to report
#  the average of running the same model N times.
#  ... but for now, we don't want it to fail since I assume we are testing each model only once.
assert (df.groupby(['Wavelet Level, J', 'Patch Size, P']).count() <= 1).all().all(), 'sanity check: no duplicates'

# generate the plot
#  note:  pivot table implicitly computes the mean when the sanity check fails
import seaborn as sns
import os
heatmap_data = df.pivot_table(col2, "Patch Size, P", "Wavelet Level, J")
fig, ax = plt.subplots(dpi=300, figsize=(6, 2.5))

baseline_score = float($baseline_score)
sns.heatmap(data=heatmap_data, cmap='PRGn', annot=True, fmt='.03f',
cbar=False, ax=ax, norm=plt.cm.colors.CenteredNorm(float(baseline_score)))
# ax.set_title('Predictive Performance: Test ROC AUC')

# save the plot to file
savefp = f'./results/plots/heatmap_perf__levels_vs_patchsize__{ns.runid_regex}.png'
if os.path.exists(savefp):
    os.remove(savefp)
ax.figure.savefig(savefp, bbox_inches='tight')
heatmap_data.to_csv(savefp.replace('.png', '.csv'))
print(f'save to: {savefp}')
EOF

ls -ltr results/plots/heatmap_perf__levels_vs_patchsize__"$runid_regex".png && echo "done" || echo "failure"
