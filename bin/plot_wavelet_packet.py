import skimage.data
import torch as T
from deepfix import plotting as P
from deepfix.models.wavelet_packet import WaveletPacket2d
from matplotlib import pyplot as plt
plt.ion()


# show different wavelets
im = T.tensor(plt.imread('data/x-ray.jpg')/255)
#  im -= im.min()
J = 2
plt.figure();plt.imshow(im, cmap='gray')
for wavelet in ['bior1.3']:
#  for wavelet in ['haar']:
#  'db3', 'coif2', 'coif1', 'bior2.2', 'bior3.1', 'rbio1.3' 'sym3', 'bior1.1', 'bior1.5'
    wp = WaveletPacket2d(wavelet, J)
    res = wp(im.unsqueeze(0).unsqueeze(0).float())
    P.plot_img_grid(res.squeeze(0).squeeze(0), suptitle=wavelet, cmap='RdYlGn')#, norm=plt.cm.colors.SymLogNorm(.1))

# show different levels
im = T.tensor(skimage.data.cell() / 255.).unsqueeze(0).unsqueeze(0).float()
for J in [1,2]:
    wp = WaveletPacket2d('bior1.3', J)
    res = wp(im)
    axs = P.plot_img_grid(res.squeeze(0).squeeze(0), suptitle=f'J={J}', cmap='RdYlGn')#, norm=plt.cm.colors.SymLogNorm(.1))
