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
    'results/1.HL1.rline_200heart2/checkpoints/epoch_100.pth', map_location=device)
    #  'results/5.HL8.hline+densenet./checkpoints/best.pth', map_location=device)
print(dct.keys())
model_name = 'RLine+Heart'
model = dct['model']
# TODO: remove temp hack for backwards compatibility with old models
model.lines_fn.sum_aggregate = False
model.lines_fn.ret_img = False
# baseline model
#  densenet = T.load('results/1.HL1.densenet/checkpoints/epoch_100.pth', map_location=device)['model']
#
# explainer
#  explainer = captum.attr.NoiseTunnel(captum.attr.InputXGradient(model.mlp))#, True))
#  explainer = captum.attr.NoiseTunnel(captum.attr.DeepLift(model.mlp))#, True))
explainer = captum.attr.NoiseTunnel(captum.attr.IntegratedGradients(model.mlp))
#  explainer2 = captum.attr.NoiseTunnel(captum.attr.IntegratedGradients(densenet.cpu()))
#  explainer2 = captum.attr.IntegratedGradients(densenet)


for i in [0,6,1,3]:
    x,y = dsets['test_dset'][i]
    print(i, y)

    x = x.unsqueeze_(0).float().to(device, non_blocking=True)
    #
    # make the privatized representation
    if model.quadtree is not None:
        repr = model.quadtree(x)
    else:
        repr = x
    repr = model.lines_fn(repr)
    #
    # get an explanation from it
    #  _img_baseline2 = T.nn.functional.interpolate(MedianPool2d(12, stride=1)(x), x.shape[-2:])
    #  _img_baseline1 = model.lines_fn(_img_baseline2)
    #  attr = explainer.attribute(repr, baselines=_img_baseline1)#, nt_samples_batch_size=2)
    #  attr = attr.detach()
    #  attr2 = explainer2.attribute(x.clone(), baselines=_img_baseline2)#, nt_samples_batch_size=2)
    #  attr2 = attr2.detach()
    attr = explainer.attribute(repr.clone().requires_grad_(True), nt_samples=20, nt_samples_batch_size=1)
    attr = attr.detach()
    #  attr2 = explainer2.attribute(x.clone().requires_grad_(True))#, nt_samples=20, nt_samples_batch_size=1)
    #  attr2 = attr2.detach()
    # --> reconstruct back to an image
    attr_img = attr.new_zeros(x.shape)
    attr_img[model.lines_fn.arr] = attr
    attr2 = MedianPool2d(24, stride=1, quantile=.90)(attr_img)
    # --> recon img
    recon_img = repr.new_zeros(x.shape)
    recon_img[model.lines_fn.arr] = repr
    #
    # get prediction
    yhat_model = model.mlp(repr).sigmoid().item()
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
    ax3.imshow(attr_img.squeeze().abs().cpu().numpy(), cmap='Blues', vmin=0)
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
