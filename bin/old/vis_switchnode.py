from waveletfix.weight_saliency import get_saliency, SaliencyResult, reinitialize_least_salient
from simplepytorch.datasets import CheXpert_Small, PreProcess
import torch as T
import math
import numpy as np


def reshape_saliency_to_img(sr:SaliencyResult):
    s1flat = T.cat([x.view(-1) for x in sr.saliency], 0
                   ).to('cpu', non_blocking=True)
    z = math.sqrt(s1flat.numel())
    h,w = int(z), int(z)+1
    assert h*w >= s1flat.shape[0]
    out = np.zeros(h*w)
    out[-s1flat.shape[0]:] = s1flat.numpy()
    return out.reshape(h,w)


def plot_switchnode(model, loader, device, num_minibatches):
    sal = get_saliency(
        lambda y, yh: (y*yh).sum(),
        model, loader, device, num_minibatches)


def plot_saliency_before_after_reinitializing():
    model = get_model(True)
    dset, loader = get_dset_loader()
    device = 'cpu'
    num_minibatches = 2  # TODO: increase

    s1 = get_saliency(lambda y, yh: (y*yh).sum(), model, loader, device, num_minibatches)

    # reinitialize one weight hammer
    #  model.fc.weight.data[0,0] = model.fc.weight.data.mean() * 1000000

    frac = .2
    reinitialize_least_salient(
        # note: calls get_saliency again...
        lambda y, yh: (y*yh).sum(), model, loader, device, num_minibatches, frac=frac, opt=None)
    frac = .15
    reinitialize_least_salient(
        # note: calls get_saliency again...
        lambda y, yh: (y*yh).sum(), model, loader, device, num_minibatches, frac=frac, opt=None)
    frac = .1
    reinitialize_least_salient(
        # note: calls get_saliency again...
        lambda y, yh: (y*yh).sum(), model, loader, device, num_minibatches, frac=frac, opt=None)
    frac = .05
    reinitialize_least_salient(
        # note: calls get_saliency again...
        lambda y, yh: (y*yh).sum(), model, loader, device, num_minibatches, frac=frac, opt=None)

    s2 = get_saliency(lambda y, yh: (y*yh).sum(), model, loader, device, num_minibatches)

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1,3, figsize=(12, 4))
    s1img = reshape_saliency_to_img(s1)  # values impossible to see...
    s2img = reshape_saliency_to_img(s2)  # values impossible to see...
    print(s2img.var())
    def normalize_for_viewing(x):
        x = x.clip(*np.quantile(x.ravel(), [0.01, .99]))
        #  return (x-x.min()) / (x.max()-x.min())
        return (x-x.mean()) / x.std()
    axs[0].imshow(normalize_for_viewing(s1img), cmap='Greens', norm=plt.cm.colors.CenteredNorm())
    axs[1].imshow(normalize_for_viewing(s2img), cmap='Greens', norm=plt.cm.colors.CenteredNorm())
    normf = lambda x: (x-x.mean())/x.std()
    axs[2].imshow(
        normalize_for_viewing(normf(normf(s1img) - normf(s2img))),
        cmap='seismic', norm=plt.cm.colors.CenteredNorm())
    titles = [
        'Saliency\nBefore re-initialization',
        'Saliency\nAfter re-initialization',
        'Difference:\nBefore (red), After (blue)'
    ]
    [ax.set_title(tit) for ax, tit in zip(axs.ravel(), titles)]
    #  fig.suptitle(f'Switch Effect: Re-initialization of {frac*100}% of parameters')
    fig.suptitle(f'Switch Effect: Re-initialization of 20% of parameters \n(iteratively re-init (20%,15%,10%,5%))')
    fig.tight_layout()

    #  fig.savefig('Figure_2.png')
    fig.savefig('Figure_1.png')
    #  axs[2].imshow((s1img - s2img))


if __name__ == "__main__":
    import torchvision.transforms as tvt
    import torchvision.models as tvm

    def preprocess_chexpert():
        x_fn = tvt.Compose([
            # pre-process the image
            tvt.CenterCrop((320,320)),
            tvt.Resize((100,100))
        ])
        def cleanup(y):
            y[y == 3] = 0  # remap missing values to negative
            return y
        y_fn = tvt.Compose([
            lambda dct: CheXpert_Small.format_labels(
                dct, labels=CheXpert_Small.LABELS_DIAGNOSTIC),
            cleanup
        ])
        return lambda dct: ( x_fn(dct['image']), y_fn(dct) )

    def get_dset_loader():
        train_dset = CheXpert_Small(
            use_train_set=True, getitem_transform=preprocess_chexpert())
        train_loader = T.utils.data.DataLoader(
            train_dset, batch_size=8, shuffle=False, num_workers=0)
        return train_dset, train_loader

    def get_model(pretrained:bool):
        model = tvm.resnet18(pretrained=pretrained)
        # tweak inputs
        model.conv1.in_channels = 1
        model.conv1.weight = T.nn.Parameter(model.conv1.weight.data[:,[0],:,:])
        # tweak outputs
        model.fc.out_features = 14
        model.fc.bias = T.nn.Parameter(model.fc.bias.data[:14])
        model.fc.weight = T.nn.Parameter(model.fc.weight.data[:14, :])
        return model


    # note: the before-after switch node vis isn't amazing
    plot_saliency_before_after_reinitializing()
    # re-initializing increases the mean and the variance though.

    def plot_reinitialization_vs_activation_magnitude():
        ...
        #  TODO
