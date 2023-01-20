"""
nerf reconstruction idea for yuhao.

step 2:  train a neural net to do an inverse wavelet packet transform.
step 3.  train a neural net to do an inverse wavelet packet transform on sparse and missing data, and incorporating spatial position information like nerf would.
"""
from waveletfix.models.wavelet_packet import WaveletPacket2d
from matplotlib import pyplot as plt
import torchvision.transforms as tvt


img = plt.imread('./data/x-ray.jpg')
img = tvt.ToTensor()(img).unsqueeze(0)
img = tvt.CenterCrop((320,320))(img)
B,C,H,W = img.shape

J = 2
wp = WaveletPacket2d('db1', J)
# create a training dataset
data = wp(img)
assert data.shape == (B,C,4**J, H/2**J, W/2**J), 'J too high'
# todo: add the h' and w' spatial position information.
#
# create the training dataset and labels for Step 2.  labels are just patches of original image
labels = (
    img
    .reshape(H//2**J, 2**J, W//2**J, 2**J)
    .permute(0,2,1,3)
    .reshape(H*W//4**J,2**J,2**J)
)
# sanity check labels
plt.imshow(labels.reshape(H//2**J, W//2**J, 2**J,2**J).permute(0,2,1,3).reshape(H,W))
# ... and for step 3...
c,hpos,wpos = T.meshgrid(T.arange(4**J), T.arange(H//2**J), T.arange(W//2**J))
# get the 4**J "wavelet scales" in last dimension
data_step2 = data.permute(0,1,3,4,2).reshape(-1,4**J)
del data
hpos = hpos.unsqueeze_(0).unsqueeze_(0).permute(0,1,3,4,2).reshape(-1,4**J)
wpos = wpos.unsqueeze_(0).unsqueeze_(0).permute(0,1,3,4,2).reshape(-1,4**J)
assert data_step2.shape == hpos.shape == wpos.shape, 'sanity check'
assert labels.shape[0] == data_step2.shape[0]
assert labels.shape[1:] == (2**J, 2**J)
# get the data for step 3
m = T.randint(0,2,(H//2**J * W//2**J, 4**J,))  # create some missing data.
m[(data_step2.abs()<1e-6) == 0] = 0
data_step3 = T.dstack([data_step2*m, m, hpos, wpos])  # todo: missing distortion info

# now train a model f(w=data_stap2[row]) on rows of data_step2 to reconstruct corresponding labels.   this just trains model to do the inverse wavelet packet transform...
# next, train model f(w=data_step3[row,:,0], m=data_step3[row,:,1], h'=data_step3[row,0,2], w'=data_step3[row,0,3], data_step3[row,:,3] to reconstruct labels.   (Step 3)
#  ... two variations:  first, learn the reconstruction directly.  second, if easy to do, the reconstruct step should just fill in missing values of the transformed data, then use an inverse wavelet packet transform.
