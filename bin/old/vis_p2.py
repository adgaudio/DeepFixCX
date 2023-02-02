from dataclasses import dataclass
from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser, choice
import math
import os
import pandas as pd
import seaborn as sns
import torch as T
import torchvision.models as tvm
import torchvision.transforms as tvt
from typing import Callable

from simplepytorch.datasets import CheXpert_Small, IntelMobileODTCervical, PreProcess
from deepfixcx.analyze_layer import analyze_model_at_modules
from deepfixcx.weight_saliency import costfn_multiclass
#  from deepfixcx.models import get_resnet, get_efficientnetv1


def get_model(model_name:str) -> T.nn.Module:
    """
    Args:
        model_name: of form "model_name:pretraining:in_channels"
            for example: model_name="resnet18:imagenet:3"
    """
    in_channels = int(model_name.rsplit(':', 1)[1])
    if model_name.startswith('resnet18:imagenet:'):
        #  model = get_resnet('resnet18', 'imagenet', in_channels, 5)
        model = tvm.resnet18(pretrained=True)
    elif model_name.startswith('efficientnet-b0:imagenetadv:'):
        #  model = get_efficientnetv1('efficientnet-b0', 'imagenetadv', 1, 5)
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0', advprop=True)
    elif model_name.startswith('resnet18:IntelMobileODT:'):
        model = T.load(
            './results/E1.resnet18.IntelMobileODT/checkpoints/epoch_50.pth'
        )['model']

    # hack to set first conv channels == 1.
    assert in_channels in {3,1}, in_channels
    if in_channels == 1:
        for x in model.modules():
            if isinstance(x, T.nn.Conv2d):
                if x.in_channels == 3:
                    x.in_channels = 1
                    x.weight.data = x.weight.data[:, [2]]
                break
    # output channels don't appear to matter for now.
    #  model._fc = T.nn.Linear(1280, 14)
    #  model.fc.out_channels = 14
    #  model.fc.weight[:] = model.fc.weight.data[:14, :, :, :]
    model.eval()
    return model, model_name

def get_loader(dset):
    # might want to reduce this to a couple images that are in the coreset
    if dset == 'CheXpert':
        dset = CheXpert_Small(
            getitem_transform=lambda dct: (
                tvt.Compose([tvt.Resize(224), tvt.CenterCrop(224)])(dct['image']),
                CheXpert_Small.format_labels(
                    dct, labels=CheXpert_Small.LABELS_DIAGNOSTIC_LEADERBOARD),
            ))
        dset = T.utils.data.Subset(dset, T.randperm(len(dset))[:5].numpy()),
    elif dset == 'ACRIMA':
        raise NotImplementedError()
    elif dset == 'IntelMobileODT':
        dset = PreProcess(
            IntelMobileODTCervical(
                'train', './data/intel_mobileodt_cervical_resized'),
            lambda xy: (xy[0].float()/255., xy[1].long()-1))
        dset = T.utils.data.Subset(dset, [0,20,699,700,1400,1404])
    else:
        raise NotImplementedError()
    loader = T.utils.data.DataLoader(dset, batch_size=1, shuffle=False)
    # or consider averaging (actually, summing) gradients
    #  loader = T.utils.data.DataLoader(
        #  T.utils.data.Subset(dset, T.randperm(len(dset))[:20]),
        #  batch_size=20, shuffle=False)
    return loader

def get_data_to_analyze(model_name:str, mode:str, dset:str):
    model, model_name = get_model(model_name)
    if mode == 'pointwise':
        modules_to_analyze = [
            n for n,x in model.named_modules()
            if isinstance(x, T.nn.Conv2d) and x.kernel_size == (1,1)]
    elif mode == 'spatial':
        modules_to_analyze = [
            n for n,x in model.named_modules()
            if isinstance(x, T.nn.Conv2d) and x.kernel_size >= (1,1)]
    else:
        raise NotImplementedError(mode)
    # select just a subset of modules
    #  modules_to_analyze = modules_to_analyze[:5]
    return model_name, analyze_model_at_modules(
        model=model,
        module_names=modules_to_analyze,
        loader=get_loader(dset),
        #  grad_cost_fn=lambda yhat, y: yhat[:,y].sum(),
        grad_cost_fn=costfn_multiclass,
        device='cuda'
    )


def plot_as_subplots(lst: list['data'], ax_titles:list[str]=None,
                     fig_axs:(plt.Figure, list[plt.Axes])=None,
                     plot_fn:Callable[[plt.Axes, 'data'], None]=lambda ax, data: ax.plot(data),
                     figsize_sidelength=3):
    """Generic way to make subplots, one per list element"""
    if fig_axs is None:
        ncol, nrow = int(math.ceil(math.sqrt(len(lst)))), int(round(math.sqrt(len(lst))))
        fig, axs = plt.subplots(nrow, ncol, figsize=(nrow*figsize_sidelength, ncol*figsize_sidelength), squeeze=False)
    else:
        fig, axs = fig_axs
    if not ax_titles:
        ax_titles = [None] * len(lst)
    for weight, title, ax in zip(lst, ax_titles, axs.ravel()):
        plot_fn(ax, weight)
        if title is not None:
            ax.set_title(title)
    for ax in axs.ravel()[1+list(axs.ravel()).index(ax):]:
        ax.axis('off')
    # fig.legend()
    fig.tight_layout()
    return fig


def cmdline_arg_parser():
    @dataclass
    class Options:
        model:str = 'resnet18:imagenet:1'  # model:pretraining:in_channels
        layertype:str = choice('pointwise', 'spatial', default='pointwise')
        dset:str = choice('CheXpert', 'IntelMobileODT', default='CheXpert')
        base_dir:str = './plots/{layertype}/{model}/{dset}'
        allplots:bool = False
    par = ArgumentParser()
    par.add_arguments(Options, 'args')
    args = par.parse_args().args
    return args


if __name__ == "__main__":
    args = cmdline_arg_parser()
    print(args)

    # save directory
    base_dir = args.base_dir.format(**args.__dict__)
    print('saving to', base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # fetch data to analyze
    model_name, data = get_data_to_analyze(args.model, args.layertype, args.dset)
    # now analyze it.

    # plot across input images for each given layer.
    def plot_pairwise_across_images():
        os.makedirs(f'{base_dir}/pairwise_imgs/', exist_ok=True)
        for name in ('input', 'output', 'weight', 'grad_input', 'grad_output', 'grad_weight'):
            for layer in data:
                lst = [getattr(x, name) for x in data[layer]]
                if name in {'input', 'grad_input'}:  # input may have multiple "inputs".  Just choose the first one.
                    if max(len(x) for x in lst) > 1:
                        print("WARNING: layer has multiple inputs.  Just using first one.  layer={layer}")
                    lst = [x[0] for x in lst]
                df = pd.DataFrame(T.stack([x.view(-1) for x in lst]).numpy().T)
                del lst
                pg = sns.pairplot(data=df, kind='hist', diag_kind='hist', diag_kws=dict(bins=200), plot_kws=dict(kde=True, bins=200))
                pg.figure.suptitle(f'{name} {layer} {model_name}')
                fp = f'{base_dir}/pairwise_imgs/{name}:{layer}.png'
                print(fp)
                pg.savefig(fp)
                plt.close(pg.figure)
                del pg, df
    if args.allplots:
        plot_pairwise_across_images()

    # heatmaps input, output, weight
    def plot_single_image():
        os.makedirs(f'{base_dir}/single_image/', exist_ok=True)
        for key in data:  # ['layer2.0.downsample.0']:
            for idx in range(len(data[key])):
                res = data[key][idx]
                w = res.weight.squeeze().numpy()
                gw = res.grad_weight.squeeze().numpy()
                i = res.input[0].abs().sum(1).squeeze().numpy()
                gi = res.grad_input[0].abs().sum(0).squeeze().numpy()
                sgi = (res.input[0].squeeze()*res.grad_input[0]).abs().sum(0)
                o = res.output.abs().sum(1).squeeze().numpy()
                go = res.grad_output.abs().sum(1).squeeze().numpy()
                sgo = (res.output*res.grad_output).abs().sum(1).squeeze().numpy()
                fig, axs = plt.subplots(3, 4, figsize=(3*2, 3*3))
                axs[0,0].set_title('weight')
                axs[0,0].imshow(w, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[1,0].set_title('grad weight')
                axs[1,0].imshow(gw, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[2,0].set_title('grad*weight')
                axs[2,0].imshow(gw*w, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[0,1].set_title('avg input')
                axs[0,1].imshow(i, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[1,1].set_title('avg grad input')
                axs[1,1].imshow(gi, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[2,1].set_title('grad*input')
                axs[2,1].imshow(sgi, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[0,2].set_title('avg output')
                axs[0,2].imshow(o, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[1,2].set_title('avg grad output')
                axs[1,2].imshow(go, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[2,2].set_title('grad*output')
                axs[2,2].imshow(sgo, cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
                axs[0,3].set_title('image')
                axs[0,3].imshow(data[key][idx].metadata['X'][0].permute(1,2,0).numpy())
                axs[1,3].axis('off')
                y = data[key][idx].metadata["y"]
                yhat = data[key][idx].metadata["yhat"]
                axs[1,3].text(-.5,.5, f"""
                Correct?  {T.allclose(yhat.argmax(1), y)}
                y={y.numpy()}
                yh={yhat.numpy()}
                """)
                axs[2,3].axis('off')
                fig.savefig(f'{base_dir}/single_image/{idx}_{key}.png', bbox_inches='tight')
    if args.allplots:
        plot_single_image()

    # plot the weights as a bigger image.
    def plot_weights():
        plot_as_subplots(
            [data[key][0] for key in data],
            plot_fn=lambda ax,x:(
                ax.set_title('test' + x.module_name)
                ,
                ax.imshow(x.weight.squeeze().numpy(),
                          cmap='RdBu', norm=plt.cm.colors.CenteredNorm(0))
            ), figsize_sidelength=5).savefig(f'{base_dir}/weights.png')
    if args.allplots:
        plot_weights()

    def _plot_centered(ax, im, cmap='RdBu'):
        ax.imshow(im, norm=plt.cm.colors.CenteredNorm(), cmap=cmap)
        ax.axis('off')
    def plot_inputs_outputs():
        img = data[key][0].metadata['X'].squeeze().permute(1,2,0).numpy()
        for key in data:
            for suptitle, lst in [
                (f'{key} input', [img]+ list(data[key][0].input[0][0].unbind())),
                (f'{key} grad input', [img]+ list(data[key][0].grad_input[0].unbind())),
                (f'{key} output', [img]+ list(data[key][0].output[0].unbind())),
                (f'{key} grad output', [img]+ list(data[key][0].grad_output[0].unbind())),]:
                fig = plot_as_subplots(lst, plot_fn=_plot_centered)
                fig.suptitle(suptitle)
                fig.savefig(f'{base_dir}/{suptitle.replace(" ", "_")}.png', bbox_inches='tight')
                plt.close(fig)
    if args.allplots:
        plot_inputs_outputs()

    def plot_activation_correlations():
        # look at how decorrelated the activations are.
        # lot of redundancy.  they aren't very well decorrelated.
        #  key = 'layer4.0.downsample.0'
        for key in data:
            fig, axs = plt.subplots(2,3, figsize=(10,10))
            fig.suptitle(f"How decorrelated are the activations of a single image? layer: {key}")
            z = data[key][0].input[0][0]
            z = z.reshape(z.shape[0], -1)
            axs[0,0].set_title(r'Input $AA^\top$ (channels)')
            _plot_centered(axs[0,0], z@z.T, 'RdYlGn')
            axs[1,0].set_title(r'$A^\top A$ (spatial positions)')
            _plot_centered(axs[1,0], z.T@z, 'RdYlGn')
            #
            z = data[key][0].output[0]
            z = z.reshape(z.shape[0], -1)
            axs[0,1].set_title(r'Output $AA^\top$ (channels)')
            #  axs[0,1].imshow(z@z.T)
            _plot_centered(axs[0,1], z@z.T, 'RdYlGn')
            axs[1,1].set_title(r'$A^\top A$ (spatial positions)')
            _plot_centered(axs[1,1], z.T@z, 'RdYlGn')
            #
            z = data[key][0].grad_weight.squeeze()
            axs[0,2].set_title(r'Grad Weights $AA^\top$ (output channels)')
            _plot_centered(axs[0,2], z@z.T, 'RdYlGn')
            axs[1,2].set_title(r'Grad Weights $A^\top A$ (input channels)')
            _plot_centered(axs[1,2], z.T@z, 'RdYlGn')
            fig.tight_layout()
            fig.savefig(f'{base_dir}/cor_activations_{key}.png', bbox_inches='tight')
    if args.allplots:
        plot_activation_correlations()
    plot_activation_correlations()


    # drill into one layer one img.  observe how:
    #  - inputs relate to outputs.
    #  - weights
    # TODO: summarize each input channel with single value.  track passing through pointwise conv
    # TODO: generate a pruned model.

"""
NOTES
Pointwise layers, pairwise across images plots:
    What's interesting:
        - pointwise convolutions center the distribution at zero and make
        images more linearly correlated.
        - the layer inputs of different images have similar shape, positive
        values, closer to zero, linearly correlated.
        - layer outputs two different images: the scatterplots look like
        gaussians centered at 0, stronger linear correlation
        - scatterplot of weight gradient comparing two images has a "spiky
        ball" pattern in some (early) pointwise layers.
        - input images are more uniformly distributed, though they show some
        patterns (and as aligned x-rays, they should!)
    What's boring:
        - gradient of input and output is gaussian.  The "cross" pattern is due
        to vanishing gradients.


    conc: doesn't tell how to initialize pointwise, but suggests some layers are iteresting and gives a little insight into what pointwise does.


    # bugs:
    # - OOM error (or otherwise segfault) on the seaborn pairplots on large layers
        # or kind='kde' or kind='hist'
        # a desirable feature, but takes forever long to run: map a kde plot to the img:  g.map_lower(sns.kdeplot, levels=4, color=".2")


shreshta:
    - expand the pairwise_across_images plots
        - images of class 1 vs class 2.  (modify the line "Subset(dset, ...)"
        - compare before and after fine-tuning
    - single image analysis
alex:
    - horizontal vs vertical components of inputs
    - also do single image analysis
    - switch plots to cervix data with cervix model

    ... maybe give shreshta learnable ghaar and see if improves perf.  shouldn't
    ... try learnable ghaar with smaller frequency in 0-1 or fixed at 1

"""
