"""
Generate and visualize hierarchical and tree-like clusterings over chexpert
classes using the VecAttn layer from DeepFix.

Assume data of form:
    results/2.C17.Cardiomegaly/checkpoints/epoch_80.pth
    results/2.C17.Pneumonia/checkpoints/epoch_80.pth
    ...
Where each checkpoint contains a DeepFixEnd2End model.
"""
import sklearn.manifold
import random
import scipy.cluster.hierarchy as sph
import scipy.spatial.distance as spd
from typing import Tuple
import scipy.sparse.csgraph as spg
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import re
import torch as T
from deepfix.models.waveletmlp import VecAttn, DeepFixEnd2End


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


def get_activations(mdl: DeepFixEnd2End, pathology):
    """
    Args:
        mdl:  A DeepFixEnd2End model with spatial attn layer, trained only on 1 class.
        pathology:  The class the model was trained on
    """
    dct, _ = get_dset_chexpert(
        train_frac=.01, val_frac=.01, small=True, labels=pathology)
    loader = list(dct['test_loader'])
    layer = mdl.mlp.spatial_attn
    activations = []
    #  assert isinstance(layer, VecAttn), 'sanity check'
    layer.register_forward_hook(
        lambda model, input, output: activations.append(output.detach()))
    imgs = []
    labels = []
    for mb in loader:
        yh = mdl(mb[0])
        y = mb[1]
        assert ((0 <= y) & (y <= 1)).all(), 'unrecognized ground truth labels - values not binary'
        labels.append(T.cat([(y>.5), (yh>0)], 1))
        imgs.append(mb[0])
    return T.cat(activations, 0), T.cat(imgs, 0), [x.numpy() for x in T.cat(labels, 0).unbind(0)]


def reconstruct(tensor, orig_img_shape:Tuple[int], wavelet, J, P, I=-1):
    iwp = WaveletPacket2d(wavelet, J, inverse=True)
    H,W = orig_img_shape
    repY, repX = H//2**J//P, W//2**J//P
    assert repY == H / 2**J / P
    z = iwp(tensor.reshape(tensor.shape[0],I,4**J, P, P)
               )\
            .repeat_interleave(repX, dim=-1).repeat_interleave(repY, dim=-2)
                   #  )
    # repeat_interleave (unpooling) outside the parenthesis is the
    # "incorrect" way but it looks better.
    return z
    #  return T.nn.Upsample(orig_img_shape)(z)



if __name__ == "__main__":
    from deepfix.train import get_dset_chexpert
    from deepfix.models.wavelet_packet import WaveletPacket2d

    orig_img_shape = (320, 320)  # original image size (320x320 imgs)
    J = 5  # wavelet level
    P = 5  # patch size
    I = 1  # num input image channels (1 for x-ray)
    wavelet = 'coif2'

    plt.ion()

    fps = glob.glob('results/2.C17*/checkpoints/epoch_0.pth')
    weight_dct = {}  # class: attn_weights
    act_dct = {}  # class: activations
    label_dct = {}  # class: (y, yh)

    for fp in fps:
        mdl = T.load(fp, map_location='cpu')['model']
        mdl.compression_mdl.wavelet_encoder.adaptive = 0
        pathology = re.search(r'/2\.C17\.(.*?)/', fp).group(1)
        weight_dct[pathology] = mdl.mlp.spatial_attn[2].vec_attn.detach().numpy().squeeze()
        act_dct[pathology], imgs, labels = get_activations(
            mdl, pathology.replace('_', ' '))
        label_dct[pathology] = labels
        del mdl
    import sys ; sys.exit()

    os.makedirs('results/plots/class_clustering', exist_ok=True)
    for pathology in act_dct:
        print(pathology)
        # mean activations reconstruction
        reconstructions = reconstruct(
            act_dct[pathology], orig_img_shape, wavelet, J, P)
        fig, axs = plt.subplots(1,2, num=1, clear=True)
        axs[0].imshow(reconstructions.mean(0).squeeze())
        axs[1].imshow(reconstruct(
            act_dct[pathology].mean(0, keepdim=True), orig_img_shape, wavelet, J, P).squeeze())
        fig.suptitle(f'{pathology}: average reconstructions.')
        axs[0].set_title('iwp(activations).mean()')
        axs[1].set_title('iwp(activations.mean())')
        fig.savefig(f'results/plots/class_clustering/{pathology}_test_mean.png', bbox_inches='tight')
        #
        # plot activation reconstructions
        color_imgs = imgs.repeat((1,3,1,1))
        color_imgs[:, [1]] += reconstructions.where(
            reconstructions>0, reconstructions.new_zeros(1))
        color_imgs[:, [0]] -= reconstructions.where(
            reconstructions<0, reconstructions.new_zeros(1))
        color_imgs = color_imgs.clip(0,1)  # suppress a warning
        from deepfix.plotting import plot_img_grid
        plt.close(1)
        fig = plot_img_grid(color_imgs.permute(0,2,3,1), suptitle=pathology, ax_titles=labels, num=1)
        N = color_imgs.shape[0]
        fig.savefig(f'results/plots/class_clustering/{pathology}_test_{N}.png', bbox_inches='tight')

        #
        # plot weight reconstructions not pretty
        #  rec = reconstruct(
        #      T.tensor(weight_dct[pathology]).unsqueeze(0),
        #      orig_img_shape, wavelet, J, P)
        #  plt.figure(num=1, clear=True)
        #  plt.imshow(rec.squeeze().numpy())

    #TODO continu
    import sys ; sys.exit()


    # some example with cell img
        #  x = skimage.data.cell()[-320:,-320:]/255
        #  fig, axs = plt.subplots(1,2)
        #  axs[0].imshow(x, 'gray')
        #  wp = WaveletPacket2d('coif2', J, inverse=False)
        #  x2 = wp(T.tensor(x).reshape(1,1,320,320).float())
        #  # l1 filtering with 5x5 patch
        #  x2 = T.conv2d(
        #      x2.reshape(1, 1024, 10, 10),
        #      T.ones((1024,1,2,2)), groups=1024, stride=2).reshape(1,1,4**J,P,P)
        #  #  x2 = x2[..., ::2, ::2]
        #  #  x2 = x2.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        #  ix = reconstruct(x2, (320,320), 'coif2', J, P, 1)
        #  axs[1].imshow(ix.squeeze(), 'gray')


    #  df = pd.DataFrame(weight_dct)
    df = pd.DataFrame({k: v.mean(0) for k,v in act_dct.items()})
    X = df.T.values
    #  X = ((df - df.mean()) / (df.max()-df.min())).T.values  # TODO test
    # TODO: multiply by the norm in ./norms ?
    dists = spd.pdist(X)
    #
    # dendrogram
    #  Z = sph.linkage(dists, metric='l1')
    #  Z[:,2] = 1 + (Z[:,2] - Z[:,2].min())/Z[:,2].std()
    #  plt.figure(figsize=(14, 10))
    #  sph.dendrogram(Z, labels=df.columns, leaf_rotation=25)
    #  plt.show()
    #
    # hierarchical clustering
    model = AgglomerativeClustering(
        n_clusters=None,
        affinity='euclidean', #'precomputed',  # l1
        distance_threshold=0,  # ensures we compute whole tree, from sklearn docs
        linkage="average",
    )
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



    # PCA
    classes = list(act_dct.keys())
    X = T.cat([act_dct[k] for k in classes], 0).numpy()
    y = np.hstack([np.ones(len(act_dct[k])) * i for i, k in enumerate(classes)])
    pca = sklearn.decomposition.PCA(2, whiten=True).fit_transform(X)
    plt.figure()
    for i in range(len(classes)):
        plt.plot(pca[y==i, 0], pca[y==i, 1], 'o', c=plt.cm.Set1(i), label=classes[i], alpha=.3)
    plt.legend()

    # something else
    g = spg.csgraph_from_dense(spd.squareform(dists))
    g = nx.from_numpy_matrix(spd.squareform(dists))
    g = nx.relabel_nodes(g, dict(zip(range(df.shape[1]), df.columns)))
    plt.figure(figsize=(14, 10))
    w = np.array([g[u][v]['weight'] for u,v in g.edges()])
    def norm01(w): return (w-w.min()) / w.ptp()
    nx.draw_networkx(g, pos=nx.circular_layout(g), edge_color=norm01(w**.75))
    plt.figure(figsize=(14, 10))
    nx.draw_networkx(nx.minimum_spanning_tree(g))


    # TSNE - per class
    classes = list(act_dct.keys())
    X = T.stack([act_dct[k].mean(0) for k in classes]).numpy()
    y = np.arange(len(classes))
    #  y = np.hstack([np.ones(len(act_dct[k])) * i for i, k in enumerate(classes)])
    tsne = sklearn.manifold.TSNE(
        learning_rate=10, init='random', metric='euclidean', square_distances=True,
    ).fit_transform(X) #spd.squareform(dists))
    #  plt.scatter(tsne[:, 0], tsne[:, 1], c=y)
    #  plt.legend()
    plt.figure()
    for i in range(len(classes)):
        plt.plot(tsne[y==i, 0], tsne[y==i, 1], 'o', c=plt.cm.gist_ncar(i/len(classes)), label=classes[i], alpha=1)
    plt.legend(loc='center')

    #  TSNE - per sample
    classes = list(act_dct.keys())
    X = T.cat([act_dct[k] for k in classes], 0).numpy()
    y = np.hstack([np.ones(len(act_dct[k])) * i for i, k in enumerate(classes)])
    tsne = sklearn.manifold.TSNE(
        learning_rate='auto', init='random', metric='euclidean', square_distances=True,
    ).fit_transform(X)
    for i in range(len(classes)):
        plt.plot(tsne[y==i, 0], tsne[y==i, 1], 'o', c=plt.cm.Spectral(i/len(classes)), label=classes[i], alpha=.7)
    plt.legend()

    # NCA - per sample
    import sklearn.neighbors
    classes = list(act_dct.keys())
    X = T.cat([act_dct[k] for k in classes], 0).numpy()
    y = np.hstack([np.ones(len(act_dct[k])) * i for i, k in enumerate(classes)])
    nca = sklearn.neighbors.NeighborhoodComponentsAnalysis(2).fit_transform(X, y)
    plt.figure()
    for i in range(len(classes)):
        plt.plot(nca[y==i, 0], nca[y==i, 1], 'o', c=plt.cm.Set1(i), label=classes[i], alpha=.3)
    plt.legend()

    #  plt.scatter(nca[:, 0], nca[:, 1], c=y, label=list(y))#, label=classes[i])
    #  plt.legend(classes)

