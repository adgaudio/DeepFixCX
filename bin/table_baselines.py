from dataclasses import dataclass
import simple_parsing as ap
import simplepytorch.plot_perf as P
import scipy.stats as ss

@dataclass
class Opts:
    dset:str = ap.choice('chexpert', 'intelmobileodt', 'kimeye', default='chexpert')

    def __post_init__(self):
        self.preprocess = None
        if self.dset == 'chexpert':
            self.plot_perf_args = '2.C27 --mode 2'
            self.val_col, self.test_col = 'val_ROC_AUC LeaderboardAVG', 'test_ROC_AUC LeaderboardAVG'
            self.preprocess = chexpert_leaderboard_avg
        elif self.dset == 'intelmobileodt':
            self.plot_perf_args = '3.E8|2.E7 --mode 2'
            self.val_col, self.test_col = 'val_ROC_AUC', 'test_ROC_AUC'
        elif self.dset == 'kimeye':
            self.plot_perf_args = '2.(K2|K0) --mode 2'
            self.val_col, self.test_col = 'val_ROC_AUC', 'test_ROC_AUC'


def chexpert_leaderboard_avg(cdf):
    cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    cols_test = [f'test_ROC_AUC {col}' for col in cols]
    cols_val = [f'val_ROC_AUC {col}' for col in cols]
    cdfs['test_ROC_AUC LeaderboardAVG'] = cdfs[cols_test].mean(1)
    cdfs['val_ROC_AUC LeaderboardAVG'] = cdfs[cols_val].mean(1)
    return cdfs



par = ap.ArgumentParser()
par.add_arguments(Opts(), dest='opts')
opts = par.parse_args().opts
# print(opts)

cdfs = P.mode_0(P.bap().parse_args(opts.plot_perf_args.split(' ')))

if opts.preprocess:
    cdfs = opts.preprocess(cdfs)
# print(cdfs)

table = cdfs.loc[cdfs.groupby(['run_id', 'filename'])[opts.val_col].idxmax()][opts.test_col]
print(table)
table2 = table.groupby('run_id').agg(['mean', 'sem'])
table2['conf95'] = table2['sem'] * ss.t(df=5, loc=0, scale=1).ppf(.95+.05/2)
table2['min_conf95'] = table2['mean'] - table2['conf95']
table2['max_conf95'] = table2['mean'] + table2['conf95']
table2['pvalue'] = ss.kruskal(*[
    table.loc[rowidxs].values
    for runid, rowidxs in table.groupby('run_id').groups.items()]).pvalue
print(table2)
