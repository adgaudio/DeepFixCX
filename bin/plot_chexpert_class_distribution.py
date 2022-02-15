import pandas as pd
from matplotlib import pyplot as plt


leaderboard = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

df = pd.read_csv("./data/CheXpert-v1.0-small/train.csv")
# add a * for leaderboard classes
df.columns = [x+("*" if x in leaderboard else "") for x in df.columns]
z = df[df.columns[-14:]].melt()
z["count"] = 1
z = z.groupby(["variable", "value"]).count().unstack("value")[
    [("count", 0.0), ("count", 1.0)]
].droplevel(None, axis=1)
ax = z.plot.barh(stacked=True)
plt.tight_layout()
plt.savefig('./results/plots/chexpert_class_distribution_TRAIN.png', bbox_inches='tight')



df = pd.read_csv("./data/CheXpert-v1.0-small/valid.csv")
df.columns = [x+("*" if x in leaderboard else "") for x in df.columns]
z = df[df.columns[-14:]].melt()
z["count"] = 1
z = (
    z.groupby(["variable", "value"])
    .count()
    .unstack("value")[[("count", 0.0), ("count", 1.0)]]
    .droplevel(None, axis=1)
)
z.plot.barh(stacked=True)
plt.tight_layout()
plt.savefig('./results/plots/chexpert_class_distribution_TEST.png', bbox_inches='tight')
