import pandas as pd
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


fps = [
    ('Compression (%)', 'results/plots/compression_ratio_varying_patch_and_level.csv'),

    # Privacy Claims:
    ('Privacy: Reconstruction', 'results/plots/heatmap_anonymity_ks_2000.csv'),
    # TODO: reconstruction ^^
    ('Privacy: Re-identification', 'results/plots/heatmap_anonymity_ks_2000.csv'),

    # Predictive Performance
    # ... BAcc
    #  ('Predictive Performance', 'results/plots/heatmap_perf__levels_vs_patchsize__2.C21.csv'),
    # ... ROC AUC
    ('Predictive Performance', 'results/plots/heatmap_perf_rocauc__2.C21.csv'),
]


df = pd.concat({name: pd.read_csv(fp) for name, fp in fps}, names=['Claim', 'ignore'])\
        .reset_index(level='ignore', drop=True)\
        .set_index('Patch Size, P', append=True)\
        .rename_axis(columns='Wavelet Level, J').stack()\
        .reset_index().rename(columns={0: 'value'})
df.head()

df2 = df.pivot(['Patch Size, P', 'Wavelet Level, J'], 'Claim', 'value').copy()
df2['Privacy: Re-identification'] = df2['Privacy: Re-identification'] ** 2
df2.head()
pd.plotting.scatter_matrix(df2)

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
# adjust size according to a sense of "depth"
vec = df2.copy()
vec['Compression (%)'] = (vec['Compression (%)']/100)**.001
#  vec = ((df2-df2.mean()) / (df2.std()))
vec = (vec.max() - vec.min()).values
#  vec = vec.mean().values
vec_dir = vec / ((vec**2).sum())**.5
depth = (df2@vec_dir.reshape(-1,1))
x, y, z = 'Compression (%)', 'Privacy: Reconstruction', 'Privacy: Re-identification'
mapp = ax.scatter(
    df2[x].values,
    df2[y].values,
    df2[z].values,
    c=df2['Predictive Performance'].values, cmap='copper',
    s=30*depth.values,
    depthshade=True
)
ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)
fig.colorbar(mapp, ax=ax, label='Predictive Performance')
