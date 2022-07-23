import torch as T
import captum.attr
from matplotlib import pyplot as plt

from heartspot.train import get_dset_chexpert
from heartspot.models.median_pooling import MedianPool2d

def clampq(x: T.Tensor, quantiles=(0,.99)):
    if isinstance(quantiles, (list, tuple)):
        quantiles = x.new_tensor(quantiles)
    return x.clamp(*x.quantile(quantiles))


dsets, _ = get_dset_chexpert(
    .9, .1, small=True, labels='Cardiomegaly', epoch_size=15000)
device = 'cuda'

# our model
dct = T.load(
    'results/5.HL8.median+hline+densenet./checkpoints/best.pth', map_location=device)
print(dct.keys())
model_name = 'HLine'
model = dct['model']
explainer = captum.attr.Saliency(model[-1])


for i in [0,6,1,3]:
    x,y = dsets['test_dset'][i]
    print(i, y)

    x = x.unsqueeze_(0).float().to(device, non_blocking=True)
    # make the privatized representation
    repr = model[:-1](x)
    # get attribution from the cardiomegaly detector (densenet)
    attr = explainer.attribute(repr.clone().requires_grad_(True)) #, nt_samples=20, nt_samples_batch_size=1)
    attr = attr.detach()
    # --> reconstruct back to an image, assuming RLine is in -2 position of Sequential.
    attr_img = attr.clone()
    heartpot_spatial_prior_mask = model[-2].arr
    attr2 = MedianPool2d(24, stride=1, quantile=.95)(attr_img)
    # --> recon img
    recon_img = repr.new_zeros(repr.shape)
    recon_img[heartpot_spatial_prior_mask] = repr[heartpot_spatial_prior_mask]

    #
    # get prediction
    #  with T.no_grad():
        #  yhat_model = model[-1](repr).sigmoid().item()
    #  yhat_densenet = densenet(x).sigmoid().item()

    fig, axs = plt.subplots(1,4, dpi=200, figsize=(4*3,3))
    [ax.axis('off') for ax in axs]
    (ax1,ax2,ax3,ax4) = axs
    #  (ax1,ax2,ax4) = axs
    ax1.set_title('Non-Privatized Image')
    ax1.imshow(x.squeeze().abs().cpu().numpy(), cmap='Greys')
    ax2.set_title('Privatized Image')
    ax2.imshow(recon_img.squeeze().cpu().numpy(), cmap='Greys')
    ax3.set_title('Attribution')
    #  ax3.imshow(attr_img.squeeze().abs().cpu().numpy(), cmap='Blues', vmin=0)
    ax3.imshow(attr_img.squeeze().cpu().numpy(), cmap='Blues')
    #  captum.attr.visualization.visualize_image_attr(
        #  T.nn.functional.interpolate(attr_img, (320,320)).squeeze().unsqueeze(-1).cpu().numpy(),
        #  x.squeeze().unsqueeze(-1).cpu().numpy(),
        #  outlier_perc=0.0001, alpha_overlay=.8,
        #  method='blended_heat_map', plt_fig_axis=(None, ax3))
    ax4.set_title('Attribution after QuantilePool')
    captum.attr.visualization.visualize_image_attr(
        T.nn.functional.interpolate(attr2, (320,320)).squeeze().unsqueeze(-1).cpu().numpy(),
        x.squeeze().unsqueeze(-1).cpu().numpy(),
        outlier_perc=0.0001, alpha_overlay=.8,
        method='blended_heat_map', plt_fig_axis=(None, ax4))
    fig.tight_layout()

    plt.show(block=False)
    fig.savefig(f'saliency_{i}.png', bbox_inches='tight')
    #  break


def tmpplot_attr(model, explainer, device):
    """Quick plot for fig1 that gives saliency attribution image (Fig. 1d)"""
    fig, ax = plt.subplots(1,1, dpi=300, )
    import torchvision.transforms as tvt
    im = T.tensor(plt.imread('./data/CheXpert-v1.0-small/train/patient64533/study1/view1_frontal.jpg')/255, dtype=T.float)
    im = tvt.CenterCrop((320,320))(im).reshape(1,1,320,320)
    x = im.to(device)
    if model.quadtree is not None:
        repr = model.quadtree(x)
    else:
        repr = x
    repr = model.lines_fn(repr)
    attr = explainer.attribute(repr.clone().requires_grad_(True), nt_samples=20, nt_samples_batch_size=1)
    attr = attr.detach()
    attr_img = attr.new_zeros(x.shape)
    attr_img[model.lines_fn.arr] = attr
    attr2 = MedianPool2d(24, stride=1, quantile=.90)(attr_img)
    ax.imshow(clampq(attr2.squeeze().abs(), (0,.99)).cpu().numpy())
    ax.axis('off')
    fig.savefig('tmpplot/attr_rhline.png', bbox_inches='tight')



tmpplot_attr(model, explainer, device=device)
