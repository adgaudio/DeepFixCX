import pandas as pd
from matplotlib import pyplot as plt


leaderboard = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def make_plot(fp_csv_in, fp_img_out):
    df = pd.read_csv(fp_csv_in)
    # add a * for leaderboard classes
    df.columns = [x+("*" if x in leaderboard else "") for x in df.columns]
    z = df[df.columns[-14:]].melt(var_name='Diagnostic Class')
    z["count"] = 1
    z = (z
         .replace(-1,2)  # set uncertain values as 2
         .fillna(-2)  # set missing values as -2
         .groupby(["Diagnostic Class", "value"]).count().unstack("value") #  [[("count", 0.0), ("count", 1.0)]
         .droplevel(None, axis=1))
    cols = {-2: '- (Missing)', 0: '-', 1: '+', 2: 'Uncertain'}
    z.columns = [cols[k] for k in z.columns]
    ax = z.plot.barh(stacked=True)
    plt.tight_layout()
    plt.savefig(fp_img_out, bbox_inches='tight')


make_plot(
    './data/CheXpert-v1.0-small/train.csv',
    './results/plots/chexpert_class_distribution_TRAIN.png')
make_plot(
    './data/CheXpert-v1.0-small/valid.csv',
    './results/plots/chexpert_class_distribution_TEST.png')
