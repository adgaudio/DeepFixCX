from dataclasses import dataclass
from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser, choice
import os
import pandas as pd
import seaborn as sns
import torch as T
import torchvision.models as tvm
import torchvision.transforms as tvt

from simplepytorch.datasets import CheXpert_Small
from deepfix.analyze_layer import analyze_model_at_modules
#  from deepfix.models import get_resnet, get_efficientnetv1


def get_model(model_name:str, in_channels=1):
    if model_name == 'resnet18:imagenet':
        #  model = get_resnet('resnet18', 'imagenet', in_channels, 5)
        model = tvm.resnet18(pretrained=True)
    elif model_name == 'efficientnet-b0:imagenetadv':
        #  model = get_efficientnetv1('efficientnet-b0', 'imagenetadv', 1, 5)
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0', advprop=True)

    # hack to set first conv channels == 1.
    for x in model.modules():
        if isinstance(x, T.nn.Conv2d):
            assert x.in_channels == 3, "hmm maybe this hack doesn't apply here"
            x.in_channels = 1
            x.weight.data = x.weight.data[:, [2]]
            break
    # output channels don't appear to matter for now.
    #  model._fc = T.nn.Linear(1280, 14)
    #  model.fc.out_channels = 14
    #  model.fc.weight[:] = model.fc.weight.data[:14, :, :, :]
    model.eval()
    return model, model_name

def get_loader():
    # might want to reduce this to a couple images that are in the coreset
    if dset == 'CheXpert':
        dset = CheXpert_Small(
            getitem_transform=lambda dct: (
                tvt.Compose([tvt.Resize(224), tvt.CenterCrop(224)])(dct['image']),
                CheXpert_Small.format_labels(
                    dct, labels=CheXpert_Small.LABELS_DIAGNOSTIC_LEADERBOARD),
            ))
    elif dset == '
    loader = T.utils.data.DataLoader(
        # would be better to hand-tune the subset to some useful images
        T.utils.data.Subset(dset, T.randperm(len(dset))[:5].numpy()),
        batch_size=1, shuffle=False)
    # or consider averaging (actually, summing) gradients
    #  loader = T.utils.data.DataLoader(
        #  T.utils.data.Subset(dset, T.randperm(len(dset))[:20]),
        #  batch_size=20, shuffle=False)
    return loader

def get_data_to_analyze(model_name:str, mode:str):
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
        loader=get_loader(),
        grad_cost_fn=lambda yhat, y: yhat.sum(),
        device='cuda'
    )


def cmdline_arg_parser():
    @dataclass
    class Options:
        model_name:str = choice('resnet18:imagenet', 'efficientnet-b0:imagenetadv', default='resnet18:imagenet')
        mode:str = choice('pointwise', 'spatial', default='pointwise')
        base_dir:str = './plots/{mode}/{model_name}'
    par = ArgumentParser()
    par.add_arguments(Options, 'args')
    args = par.parse_args().args
    return args


if __name__ == "__main__":
    args = cmdline_arg_parser()

    # save directory
    base_dir = args.base_dir.format(**args.__dict__)
    print('saving to', base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # fetch data to analyze
    model_name, data = get_data_to_analyze(args.model_name, args.mode)
    # now analyze it.

    # plot across input images for each given layer.
    for name in ('input', 'output', 'weight', 'grad_input', 'grad_output', 'grad_weight'):
        for layer in data:
            lst = [getattr(x, name) for x in data[layer]]
            if name in {'input', 'grad_input'}:  # input may have multiple "inputs".  Just choose the first one.
                if max(len(x) for x in lst) > 1:
                    print("WARNING: layer has multiple inputs.  Just using first one.  layer={layer}")
                lst = [x[0] for x in lst]
            df = pd.DataFrame(T.stack([x.view(-1) for x in lst]).numpy().T)
            del lst
            pg = sns.pairplot(data=df, kind='hist', diag_kind='kde', plot_kws=dict(kde=True, bins=200))
            pg.figure.suptitle(f'{name} {layer} {model_name}')
            fp = f'{base_dir}/pairplot:{name}:{layer}.png'
            print(fp)
            pg.savefig(fp)
            plt.close(pg.figure)
            del pg, df

    # drill into one layer.  observe how:
    #  - inputs relate to outputs.
    #  - weights

    # - activations of pointwise layers are VERY similar across samples, maintaining a strong linear almost 1:1 relationship
    #  - the input images themselves are more uniformly distributed, though they show some patterns (and as aligned x-rays, they should!)
    #  - the pre-activations of different images have remarkably similar shape (very strong linear correlation, but varying (positive) scales)
    #  - the activations of different images have even stronger linear correlation.  They are shifted to be centered at zero and they are more symmetric
    # - gradients are weirdly correlated to each other, giving very clear repetitive patterns ("spiky ball").  The (max) number of spikes seems about the same, but direction varies.
    # -

    # bugs: using ModuleDataPoint when shouldnt
    # map a kde plot to the img:  g.map_lower(sns.kdeplot, levels=4, color=".2")
    # or kind='kde' or kind='hist'
