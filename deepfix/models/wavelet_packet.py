import pywt
from typing import Union
import warnings
import numpy as np
import torch as T


class WaveletPacket2d(T.nn.Module):
    def __init__(self, wavelet:str, levels:Union[int,str]):
        super().__init__()
        lo, hi = pywt.Wavelet(wavelet).filter_bank[:2]
        filters = T.tensor(np.stack([
            np.outer(lo, lo),
            np.outer(lo, hi),
            np.outer(hi, lo),
            np.outer(hi, hi)]), dtype=T.float).unsqueeze(1)  # shape: O, I, H_k, W_k
        self.register_buffer('filters', filters)
        self.levels = levels
        self.conv_params = dict(
            stride=(2,2),  #filters[0,0].shape,
            padding=tuple(np.array(filters[0,0].shape)//2),
            dilation=1,
            bias=None
        )

    def get_max_level(self, input_shape:tuple[int]) -> int:
        H, W = input_shape[2:]
        s_h, s_w = self.conv_params['stride']
        J_max = min(  # solve for J the eqtn:  1 = H / stride**J
            int(np.log2(H/s_h) / np.log2(s_h)),
            int(np.log2(W/s_w) / np.log2(s_w)),)
        return J_max

    def forward(self, x:T.Tensor) -> T.Tensor:
        """
        Transform input images.
        Args:
            x: tensor of shape (B,C,H,W)
        Returns:
            Tensor of shape: (B,C,D,H',W') where
              - D is the number of detail matrices, D=4**J for J wavelet levels
              - H' and W' are the size of the spatial dimension after transform
                H/2**(J-1) >= H' >= H/2**J  ... and similarly for W'
        """
        B, I, H, W = x.shape
        sanity_check_num_elems = x.numel()
        tmp = x
        J = self.levels
        if J == 'max':
            J = self.get_max_level(x.shape)
        elif J > (J_max := self.get_max_level(x.shape)):
            warnings.warn(
                f'The max wavelet levels for input data of shape {x.shape}'
                f' is {J_max}, but we have levels={J}.'
                '  Decrease levels or use smaller wavelet filters to avoid'
                ' wasting computation and ram usage.')
        for lvl in range(J):
            if lvl > 0:
                tmpfilters = tmpfilters.repeat(4,1,1,1)
            else:
                tmpfilters = self.filters.repeat(tmp.shape[1],1,1,1)
            #  print('a', tmp.shape)
            tmp = T.conv2d(tmp, tmpfilters, groups=tmp.shape[1],
                           **self.conv_params)
            #  print('b', tmp.shape)
        return tmp.reshape(B, I, -1, *tmp.shape[-2:])


if __name__ == "__main__":

    import skimage.data
    from deepfix import plotting as P
    from matplotlib import pyplot as plt
    im = T.tensor(skimage.data.cell() / 255.).unsqueeze(0).unsqueeze(0).float()
    #  im = T.tensor(np.random.randn(512, 512)).unsqueeze(0).unsqueeze(0).float()
    J = 3
    wp = WaveletPacket2d('db1', J)
    res = wp(im)
    res = res.reshape(1,-1,*res.shape[-2:])
    print(res.shape)
    plt.ion()
    plt.figure() ; plt.imshow(im.squeeze(), cmap='gray')
    P.plot_img_grid(res[0])#, norm=plt.cm.colors.SymLogNorm(.001))
    #  visualize the sum of coefficients
    pos_coeffs = res.where(res > 0, T.tensor(0.))
    neg_coeffs = res.where(res < 0, T.tensor(0.))
    fig, (ax1, ax2) = plt.subplots(2,1)
    fig.suptitle(f'J={J}')
    ax1.plot(pos_coeffs[0,1:].sum((-1,-2)))
    ax1.set_title('sum of pos coefficients per matrix')
    ax2.plot(neg_coeffs[0,1:].sum((-1,-2)))
    ax2.set_title('sum of neg coefficients per matrix')
    fig.tight_layout()

    # test max level
    wp = WaveletPacket2d('coif1', 'max')
    print(wp(im).shape)
    assert wp.get_max_level((None, None, 512, 512)) == 8
    assert wp(im).shape == (1, 1, 4**8, 4, 4)  # where coif1 has 6 filters, so h<6 and w<6

    # test that wavelet packet outputs correct num channels
    J=8
    wp = WaveletPacket2d('coif1', J)
    im2 = T.tensor(skimage.data.camera() / 255.).unsqueeze(0).unsqueeze(0).float()
    imc = T.cat([im[:,:,:512, :512], im2], 1)
    res2 = wp(imc)
    res2 = res2.reshape(1,-1,*res2.shape[-2:])
    b,c,h,w = res2.shape
    assert b == 1
    assert c == 2*(4**J)
    assert 512/2**(J-1) > h > 512/2**J
    assert 512/2**(J-1) > w > 512/2**J
    # test that wavelet packet doesn't mix channels
    wp = WaveletPacket2d('db1', 2)
    res3 = wp(imc)
    res3 = res3.reshape(1,-1,*res3.shape[-2:])
    P.plot_img_grid(res3[0])  # check that two images aren't mixed.

    # check that channels are properly separated from each other.  no mixing.
    wp = WaveletPacket2d('db1', 2)
    imd = T.stack([
        T.ones((100,100)),
        T.zeros((100,100))]).unsqueeze(0)
    res4 = wp(imd)
    P.plot_img_grid(res4.reshape(-1,*res4.shape[-2:]))
    print(res4.shape)
    assert res4[0,1].abs().sum() == 0
    assert res4[0,0].abs().sum() > 1

