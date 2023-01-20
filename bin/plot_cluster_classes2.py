from collections import defaultdict
from itertools import product
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from typing import Tuple
import captum.attr
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import scipy.cluster.hierarchy as sph
import torch as T
from waveletfix.train import get_dset_chexpert
from waveletfix.models.wavelet_packet import WaveletPacket2d
from waveletfix import plotting as dplt


def get_linkage_matrix_from_sklearn(model):
    # Create linkage matrix
    #  https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    #
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    #
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


def linkage_to_networkx(linkage_matrix, labels:Tuple):
    labels = dict(zip(range(len(labels)), labels))
    G = nx.Graph()
    n = len(linkage_matrix)
    for i in range(n):
        node1 = linkage_matrix[i, 0]
        node1 = labels.get(node1, node1)
        node2 = linkage_matrix[i, 1]
        node2 = labels.get(node2, node2)
        weight = linkage_matrix[i, 2]
        cluster_idx = i+n+1
        G.add_edge(node1, cluster_idx, weight=weight)
        G.add_edge(node2, cluster_idx, weight=weight)
    return G


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.

    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
            G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0,
            xcenter = 0.5, rootpos = None, leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for i,child in enumerate(children):
                nextx += rootdx
                #  dy = G[root][child].get('weight',1)
                rootpos, leafpos, newleaves = _hierarchy_pos(
                    G,child, leftmost+leaf_count*leafdx,
                    width=rootdx, leafdx=leafdx,
                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap + i/len(children)*vert_gap,
                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(
        G, root, 0, width, leafdx=width*1./leafcount, vert_gap=vert_gap,
        vert_loc=vert_loc, xcenter=xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1])
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos



if __name__ == "__main__":

    print("setup")
    # cmdline modifiable settings
    import dataclasses as dc

    @dc.dataclass
    class CmdlineOpts:
        J:int = 2
        P:int = 19
    from simple_parsing import ArgumentParser
    args = ArgumentParser()
    args.add_arguments(CmdlineOpts, dest='args')
    args = args.parse_args().args
    # "static" config
    J, P = args.J, args.P
    wavelet='db1'
    fp = f'results/2.C21.J={J}.P={P}/checkpoints/epoch_80.pth'
    assert wavelet == 'db1' and '2.C21' in fp, 'setup error'
    device = 'cuda'
    orig_img_shape = (320, 320)  # original image size (320x320 imgs)
    B = int(os.environ.get('batch_size', 15))
    I = 1  # num input image channels (1 for x-ray)

    # get model and dataloader
    print("get model and dataloader")
    mdl = T.load(fp)['model']
    # backwards compatibility with earlier version of waveletfix
    mdl.compression_mdl.wavelet_encoder.adaptive = 0

    # get the thresholds (from the get_rocauc script) for true prediction.
    class_thresholds = pd.read_csv(
        f'{os.path.dirname(os.path.dirname(fp))}/class_thresholds.csv',
        index_col=0)['0']

    # VAL dataset for more samples
    dset_dct, class_names = get_dset_chexpert(.9, .01, small=True)
    d = dset_dct['val_dset']
    # create dataloader for ONLY frontal imgs
    lcsv = d.dataset.labels_csv
    idxs = np.arange(len(d.dataset))[((lcsv['index'].isin(d.indices)) & (lcsv['Frontal/Lateral'] == 0)).values]
    d = T.utils.data.Subset(d.dataset, idxs)
    N = len(d)
    d = T.utils.data.DataLoader(d, batch_size=B, shuffle=False)
    print(f'Analyzing {N} samples')

    # TEST dataset
    #  dset_dct, class_names = get_dset_chexpert(.9, .1, small=True)
    #  d = dset_dct['test_dset']
    #  lcsv = d.labels_csv
    #  #  d = dset_dct['val_dset']
    #  #  lcsv = d.dataset.labels_csv
    #  #  create dataloader for ONLY frontal imgs
    #  idxs = np.arange(len(d))[(lcsv['Frontal/Lateral'] == 0).values]
    #  d = T.utils.data.Subset(d, idxs)
    #  N = len(d)
    #  d = T.utils.data.DataLoader(d, batch_size=B, shuffle=False)
    #  import sys ; sys.exit()


    # get saliency
    print("Get Saliency of all images in dataset")
    explainer = captum.attr.DeepLift(mdl.mlp, True)
    attrs_img_mean, attrs_enc_mean = {}, {}
    attrs_cm = defaultdict(lambda: (T.tensor(0.), 0))
    gtlabels, preds = [], []
    for mb in d:
        x = mb[0].to(device, non_blocking=True)
        _B = x.shape[0]
        # get Deepfix encoding
        enc = mdl.compression_mdl(x)
        # Use the approximation image as baseline for the attribution method
        baseline = enc.reshape(_B, 4**J,P,P).clone()
        baseline[:, 1:] = 0
        baseline = baseline.reshape(enc.shape)
        # save predictions / ground truth
        gtlabels.append(mb[1])
        with T.no_grad():
            preds.append(mdl.mlp(enc).cpu())
        # reconstruct the image
        recon_img = mdl.compression_mdl.reconstruct(
            enc.detach(), orig_img_shape, wavelet=wavelet, J=J, P=P).cpu()
        # Compute attribution maps.  Assume no access to original img
        attrs_enc, attrs_img = {}, {}
        for i, kls in enumerate(class_names):
            attrs_enc[kls] = (
                explainer.attribute(
                    enc, target=i, baselines=baseline,
                    #  nt_samples=15, nt_samples_batch_size=B, nt_type='smoothgrad',
                )
                .detach().reshape((_B, I, 4**J, P, P)).float()) ** 2
            attrs_img[kls] = mdl.compression_mdl.reconstruct(
                attrs_enc[kls], orig_img_shape, wavelet=wavelet, J=J, P=P,
                restore_orig_size=True).cpu().abs()
        # Aggregate attribution maps
        for i, kls in enumerate(class_names):
            # overall avg saliency per class.
            if kls not in attrs_img_mean:
                attrs_img_mean[kls] = attrs_img[kls].abs().sum(0).cpu().clone()/N
                attrs_enc_mean[kls] = attrs_enc[kls].abs().sum(0).cpu().clone()/N
            else:
                attrs_img_mean[kls] += attrs_img[kls].abs().sum(0).cpu().clone()/N
                attrs_enc_mean[kls] += attrs_enc[kls].abs().sum(0).cpu().clone()/N
            # avg saliency per class per errtype (errtypes in {fp, tp, fn, fp})
            # ... for given class, assign samples to each possible value of the
            #     confusion matrix.  aggregate by maintaining a running mean
            t = class_thresholds[kls]  # TODO: check it is correct.
            mb_stats = dict(
                tp=(mb[1][:, i] == 1) == (preds[-1][:, i] > t).cpu(),
                fp=(mb[1][:, i] == 0) == (preds[-1][:, i] > t).cpu(),
                fn=(mb[1][:, i] == 1) == (preds[-1][:, i] < t).cpu(),
                tn=(mb[1][:, i] == 0) == (preds[-1][:, i] < t).cpu())
            # running mean
            for errtyp in 'tn', 'fp', 'fn', 'tp':
                oldval, oldcnt = attrs_cm[(kls, errtyp)]
                mask = mb_stats[errtyp]
                if mask.sum() == 0: continue
                newcnt = oldcnt + mask.sum()
                _newvals = attrs_img[kls][mask].sum(0).cpu().clone()
                attrs_cm[(kls, errtyp)] = (oldval * oldcnt/newcnt + _newvals/newcnt, newcnt)
    attrs_cm = dict(attrs_cm)
    gtlabels = T.cat(gtlabels, 0)
    preds = T.cat(preds, 0)

    # plot the image and reconstruction
    dplt.plot_img_grid(x.squeeze(1), cmap='gray')
    dplt.plot_img_grid(recon_img.squeeze(1), cmap='gray')

    #
    # plot reconstructions and saliency for just a few images
    #
    for class_idx, class_name in enumerate(class_names[:4]):
        with T.no_grad():
            yhat = ((mdl.mlp(enc)[:, class_idx]>class_thresholds[class_name])*1.).tolist()
        labels = list(zip(mb[1][:, class_idx].tolist(), yhat))
        #  dplt.plot_img_grid(
            #  attrs_img[class_name].abs().squeeze(1), vmin='min', vmax='max',
            #  suptitle=f'{class_name}  (y, $\hat y$)',
            #  ax_titles=labels
        #  )
        color_imgs = recon_img.repeat((1,3,1,1)).numpy()
        color_imgs = (color_imgs - color_imgs.min()) / (color_imgs.max() - color_imgs.min())
        s = attrs_img[class_name]
        s = s.abs()  # magnitude of saliency
        s = s.cpu().numpy()
        # normalize for overlaying on top of the reconstructed image.
        s = np.abs(((s - s.mean((-1,-2), keepdims=True)) / s.std((-1,-2), keepdims=True)))
        color_imgs[:, [0]] += s
        color_imgs[:, :] /= np.percentile(color_imgs[:, [0]], 99)
        fig = dplt.plot_img_grid(color_imgs.transpose(0,2,3,1), suptitle=class_name, ax_titles=labels)


    # plot the average saliency per class
    #  fig, = dplt.plot_img_grid([attrs_cm[(k, 'fp')][0].squeeze(0).abs()+  attrs_cm[(k, 'fn')][0].squeeze(0).abs() for k in class_names], ax_titles=class_names, suptitle='Per-class Average Saliency Attribution', cmap='gray'), #, norm=plt.cm.colors.PowerNorm(.2))
    asterisk = list(sorted(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']))
    not_asterisk = list(sorted([k for k in class_names if k not in asterisk]))
    fig, = dplt.plot_img_grid(
        [attrs_img_mean[k].squeeze(0).abs().squeeze(0) for k in asterisk + not_asterisk],
        ax_titles=[(k.replace("Enlarged Car", "Enlarged\nCar")+("*" if k in asterisk else ""))
                   for k in (asterisk+not_asterisk)],
        cmap='gray', rows_cols=(2,7)), #, norm=plt.cm.colors.PowerNorm(.2))
    #  fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig('./results/plots/attribution_all_classes.png', bbox_inches='tight')
    # ... one figure of subplots with avg saliency over the confusion matrix: TP, FP, FN, TN

    #  for i, k in enumerate(class_names):
    #      arrs = []
    #      errtypes = 'tp', 'tn', 'fp', 'fn'
    #      for errtyp in errtypes:
    #          attr_map, n = attrs_cm[(k, errtyp)]
    #          print(k, errtyp, n, attr_map.shape)
    #          arrs.append(attr_map.squeeze() if n != 0 else [[np.nan]])
    #      dplt.plot_img_grid(arrs, ax_titles=errtypes, suptitle=k, rows_cols=(1,4), cmap='gray')

    # how classes distribute across wavelet level and wavelet a,h,v,d orientations.
    df = pd.DataFrame({k: attrs_enc_mean[k].abs().sum((-2,-1)).squeeze(0).cpu().numpy() for k in class_names})
    df.index = list(''.join(x) for x in product(*['AHVD']*J))
    z = df.copy()
    z = z.abs()  # tmp
    #  z = z / z.sum(1).values.reshape(-1,1)
    z = z / z.sum(0)
    ax = z.plot.bar(stacked=True, legend=False)
    ax.legend(ncol=1, loc='upper right')
    ax.figure.savefig('./results/plots/attribution_magnitude_scales_orientations.png', bbox_inches='tight')

    df = pd.DataFrame({k: v.cpu().numpy().ravel() for k, v in attrs_enc_mean.items()})
    X = pd.DataFrame({k: v.cpu().numpy().ravel() for k, v in attrs_img.items()}).T.values
    #  X = X/X.sum(1, keepdims=True)
    model = AgglomerativeClustering(
        n_clusters=None,
        affinity='euclidean',
        distance_threshold=0,  # ensures we compute whole tree, from sklearn docs
        linkage="ward",
    )
#  [
        #  'Cardiomegaly', 'Lung Opacity', 'Edema', 'Pneumothorax',
        #  'Pleural Effusion', 'Support Devices']]
    model.fit(X)
    Z = get_linkage_matrix_from_sklearn(model)
    Z[:,2] = 1 + (Z[:,2] - Z[:,2].min())/(Z[:,2]-Z[:,2].min()).std()
    G = linkage_to_networkx(Z, df.columns)
    plt.figure(figsize=(24, 12))
    pos = hierarchy_pos(G, leaf_vs_root_factor=1)
    nx.draw_networkx(G, pos=pos)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={e: round(G.edges[e]['weight'], 4) for e in G.edges},
    )
    plt.figure(figsize=(14, 10))
    sph.dendrogram(Z, labels=df.columns, leaf_rotation=25)
    #
    #
    grp = nx.community.greedy_modularity_communities(G, 'weight')
    print('modularity', '\n'.join(str(x) for x in grp))
    # inconsistent:  list(nx.community.asyn_lpa_communities(G, 'weight'))
    #
    # flatten the dendrogram into a better hierarchical structure
    G2 = G.copy()
    for grp in list(nx.community.naive_greedy_modularity_communities(G)):
        # find the node that connects the group to the tree
        parent = [x for x in grp if set(G2[x]).difference(grp)]
        if len(parent)>1:
            print(grp)
            root = parent[0]  # save this to visualize with the hierarchy layout
            continue
        assert len(parent) == 1, 'sanity check: assume each group has 1 node that connects the group to the tree'
        parent = parent[0]
        # connect the group to this node
        for child in list(grp):
            if child == parent: continue
            if isinstance(child, int):
                G2.remove_node(child)
            else:
                # connect this node only to the parent
                G2.remove_edges_from(list(G2.edges(child))) # remove all connections
                G2.add_edge(parent, child)
    plt.figure(figsize=(24,8))
    pos = hierarchy_pos(G2, root=root, leaf_vs_root_factor=1, width=1)
    nx.draw_networkx(G2, pos=pos)
