#!/usr/bin/env bash
#
# Generate a heatmap of the accuracy performance of the blur and median competing methods
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

python -m simplepytorch.plot_perf $runid_regex --mode 0 <<EOF

# select the subset of data we want to visualize.
col1 = 'val_ROC_AUC'
col2 = 'test_ROC_AUC'

if (
        os.environ.get('food101', False) == 'true'
        or os.environ.get('flower102', False) == 'true'):
    col1 = 'train_Acc'
    col2 = 'test_Acc'
if os.environ.get('chexpert', False) == 'true':
    cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    cols_val = [f'val_ROC_AUC {x}' for x in cols]
    cols_test = [f'test_ROC_AUC {x}' for x in cols]
    cdfs['val_ROC_AUC AVG'] =  cdfs[cols_val].mean(1) # the leaderboard average, not the 14 classes average (because test set is too small)
    cdfs['test_ROC_AUC AVG'] =  cdfs[cols_test].mean(1) # the leaderboard average, not the 14 classes average (because test set is too small)
    col1 += ' AVG'
    col2 += ' AVG'

if 'filename' in cdfs.index.names:
    df = cdfs.loc[cdfs.groupby(['run_id', 'filename'])[col1].idxmax()].reset_index('epoch').groupby('run_id')[['epoch', col2]].mean().sort_values(col2)
    print(df)
else:
    df = cdfs.loc[cdfs.groupby(["run_id"])[col1].idxmax()].sort_values(col2).reset_index()

# regex = 'J=(?P<J>\d+).P=(?P<P>\d+)'
# regex = r'(K=(?P<K>\d+)|J=(?P<J>\d+).P=(?P<P>\d+))'
# model_hyperparams = df.reset_index()["run_id"].str.extractall(regex).rename(columns={'J': 'Wavelet Level J', 'P': 'Patch Size P', 'K': 'Kernel size'}) #.astype('int')
# df = df.reset_index().join(model_hyperparams.reset_index('match', drop=True), how='outer')
df = df.reset_index()

savefp = "./results/plots/${runid_regex}_perf.csv"
df.to_csv(savefp, index=False)

print('test performance of the model with highest val performance (spanning all epochs and models)')
print(cdfs[col1].idxmax())
print(cdfs[col2].loc[cdfs[col1].idxmax()])

print(f'save to: {savefp}')
EOF

ls -ltr results/plots/${runid_regex}_perf.csv && echo "done" || echo "failure"
