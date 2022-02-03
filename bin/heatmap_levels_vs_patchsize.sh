#!/usr/bin/env bash
#
# Generate a heatmap of the wavelet levels vs patch size  for waveletmlp models.
# expects experiment ids to contain substring like "waveletmlp:300:1:14:4:3:1:2"
#
# use like this    $  ./this_script.sh  "experiment_id_regex"
# 
# Note: it's a cobbled together script.  If there are any errors, they will
# appear in console output, but the bash exit code isn't reliable.
#

python -m simplepytorch.plot_perf $* --mode 0 <<EOF
# select the subset of data we want to visualize.
# WARNING: the query epoch>N  step implies we are averaging over epochs >N  (like an ensemble approach).
df = cdfs\
        .query('epoch > 40')\
        .groupby(["run_id", 'filename'])["test_MCC AVG"]\
        .mean().reset_index()

# HACK to get my test to work:  REMOVE THIS!
# df.iloc[0, 0] = "1.waveletmlp:300:1:14:4:3:1:2"
# df.iloc[1, 0] = "1.waveletmlp:300:1:14:2:32:1:2"

# extract the model hyper parameters from the experiment id and join to the data we want to visualize.
# ASSUMPTION: the experiment id contains a string like "waveletmlp:300:1:14:4:3:1:2"
cols = "mlp_channels, in_ch, out_ch, wavelet_levels, patch_size, in_ch_mul, mlp_depth".split(", ")
regex = "waveletmlp:" + ":".join(f"(?P<{col}>\d+)" for col in cols)
model_hyperparams = df["run_id"].str.extractall(regex)
assert model_hyperparams.shape[0] > 0, 'sanity check: the experiments ids should contain a string like "waveletmlp:1:14:4:3:1:2"'
df = df.join(model_hyperparams.reset_index('match', drop=True), how='outer')

# you can disable this sanity check if it fails.  Failure would mean you tested
# two models with wavelet_levels=A and patch_size=B for some (A,B)
#  ... at some point, we want to make this fail because we will want to report
#  the average of running the same model N times.
#  ... but for now, we don't want it to fail since I assume we are testing each model only once.
assert (df.groupby(['wavelet_levels', 'patch_size']).count() <= 1).all().all(), 'sanity check: no duplicates'

# generate the plot
#  note:  pivot table implicitly computes the mean when the sanity check fails
import seaborn as sns
import os
ax = sns.heatmap(data=df.pivot_table("test_MCC AVG", "wavelet_levels", "patch_size"), cmap='RdYlGn')

# save the plot to file
savefp = f'./results/plots/heatmap_levels_vs_patchsize__{ns.runid_regex}.png'
os.remove(savefp)
ax.figure.savefig(savefp, bbox_inches='tight')
print(f'save to: {savefp}')
EOF

ls -ltr results/plots/heatmap_levels_vs_patchsize__"$1".png && echo "done" || echo "failure"
