from matplotlib import pyplot as plt
import torch as T
import numpy as np
from typing import Tuple, Iterable, Optional, Dict


def tolist(val, N):
    if not isinstance(val, (list, tuple)):
        val = [val] * N
    return val


def plot_img_grid(imgs: Iterable, suptitle:str = '', rows_cols: Tuple = None,
                  norm=None, vmin=None, vmax=None, cmap=None,
                  convert_tensor:bool=True, num:int=None,
                  ax_titles:Tuple[str]=None):
    """Plot a grid of n images

    :imgs: a numpy array of shape (n,h,w) or a list of plottable images
    :suptitle: figure title
    :convert_tensor: if True (default), try to convert pytorch tensor
        to numpy.  (Don't try to convert channels-first to channels-last).
    :vmin: and :vmax: and :norm: are passed to ax.imshow(...).  if vmin or vmax
        equal 'min' or 'max, respectively, find the min or max value across all
        elements in the input `imgs`.  vmin, vmax or norm can also each be a
        list, with one value per image in the figure.
    :cmap: a matplotlib colormap
    :num: the matplotlib figure number to use
    """
    if rows_cols is None:
        _n = np.sqrt(len(imgs))
        rows_cols = [int(np.floor(_n)), int(np.ceil(_n))]
        if np.prod(rows_cols) < len(imgs):
            rows_cols[0] = rows_cols[1]
    elif rows_cols[0] == -1:
        rows_cols = list(rows_cols)
        rows_cols[0] = int(np.ceil(len(imgs) / rows_cols[1]))
    elif rows_cols[1] == -1:
        rows_cols = list(rows_cols)
        rows_cols[1] = int(np.ceil(len(imgs) / rows_cols[0]))
    assert np.prod(rows_cols) >= len(imgs), (rows_cols, len(imgs))
    if vmin == 'min':
        vmin = min([x.min() for x in imgs])
    if vmax == 'max':
        vmax = max([x.max() for x in imgs])
    if convert_tensor:
        imgs = (x.cpu().numpy() if isinstance(x, T.Tensor) else x for x in imgs)
    fig, axs = plt.subplots(
        *rows_cols, squeeze=False, figsize=np.multiply((rows_cols[1],rows_cols[0]),2),
        tight_layout=dict(w_pad=0.1, h_pad=0.1), num=num, clear=True)
    #  fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0., wspace=0.)
    fig.suptitle(suptitle)
    [ax.axis('off') for ax in axs.ravel()]
    if ax_titles is None:
        ax_titles = [None] * len(fig.axes)
    norm = tolist(norm, len(fig.axes))
    vmin = tolist(vmin, len(fig.axes))
    vmax = tolist(vmax, len(fig.axes))
    for zimg, ax, ax_title, norm, vmin, vmax in zip(imgs, axs.flatten(), ax_titles, norm, vmin, vmax):
        ax.imshow(zimg, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(ax_title)
    return fig


def arrow_with_text_in_middle(
        text, left_xy, right_xy, text_xy=None, arrowstyle=('->', '->'),
        arrowprops:Optional[Dict]=None, fontsize='xx-large', ax=None,
        **text_kwargs):
    """Draw arrow like   <----  TEXT  ---->  where
         - fontsize controls both the size of arrow and the text
         - arrowstyle defines the arrow on left and right sides.
           NOTE:  cannot pass 'arrowstyle' as a key in arrowprops.
        - text_xy by default places text at midpoint.

    For one-directional arrow, could also do left_xy=None or right_xy=None,
    giving:  TEXT ---->    or  <---- TEXT
    For an arrow like <---- TEXT ----  you can define arrowstyle=('->', '-')
    """
    if text_xy is None:
        text_xy = ( (left_xy[0]+right_xy[0])/2, (left_xy[1]+right_xy[1])/2 )
    _text_kwargs = dict(
        horizontalalignment='center', verticalalignment='center', fontsize=fontsize)
    _text_kwargs.update(text_kwargs)
    text_kwargs = _text_kwargs
    text_kwargs.update(text_kwargs)
    if arrowprops is None:
        arrowprops = {}
    if ax is None:
        ax = plt
    if left_xy:
        ax.annotate(
            text, xy=left_xy, xytext=text_xy,
            arrowprops={'arrowstyle': arrowstyle[0], **arrowprops}, **text_kwargs)
        alpha = 0
    else:
        alpha = 1
    if right_xy:
        ax.annotate(
            text, xytext=text_xy, xy=right_xy,
            arrowprops={'arrowstyle': arrowstyle[1], **arrowprops},
            alpha=alpha, **text_kwargs)
