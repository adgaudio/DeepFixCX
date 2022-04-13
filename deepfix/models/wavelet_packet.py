from typing import Union
import math
import numpy as np
import pywt
import torch as T
import warnings


class WaveletPacket2d(T.nn.Module):
    """Compute a multi-level Wavelet Packet Transform or its inverse.

    wavelet:  Any wavelet supported by pywt library, or a string like
        "kaiming_normal_,N" to denote any initialization function in T.nn.init
        to be applied to a tensor of size N or of size NxN.  If using a pytorch
        initialization function, you probably want to define adaptive!=0 too.
    levels:  How many wavelet levels to compute the transform for.
      Pass an integer or the string 'max'.
    inverse:  If True, compute an inverse wavelet packet transform.
        Assumes that adaptive=0 and wavelet is one supported by pywt library
    adaptive: a value in {0,1,2}.  Determines whether to the 2d filters or 1d
        lo and hi pass filters mother wavelet are learnable (requires_grad=True)
      - If adaptive=0, no learning anything.  Just do wavelet transform.
      - If adaptive=1, allow the lo and hi pass 1-d vectors that generate the
        2d wavelet filters to be learned.  Initialized with values from
        `wavelet` . Adaptive=1 allows the wavelet packet to violate necessary
        conditions for it to be a wavelet.  E.g. to preserve invertibility, add
        a loss term that ensures this.
      - If adaptive=2, initialize 2d filters with wavelet values,
        then register them as learnable parameters, like a regular convolution.
    """
    def __init__(self, wavelet:str, levels:Union[int,str],
                 inverse:bool=False, adaptive:int=0):
        super().__init__()
        lo, hi, init_fn = self._get_1d_lo_and_hi(wavelet, inverse)

        if adaptive == 1:
            if init_fn:
                lo, hi = init_fn(lo), init_fn(hi)
            self.lo, self.hi = T.nn.Parameter(lo), T.nn.Parameter(hi)
        elif adaptive == 0 or adaptive == 2:
            self.lo, self.hi = lo, hi
            filters = self.compute_filters_2d(lo, hi)
            if init_fn:
                filters = init_fn(filters)
            if adaptive == 0:
                self.register_buffer('filters', filters)
            else:
                self.filters = T.nn.Parameter(filters)
        else:
            raise NotImplementedError(f'adaptive={adaptive}')
        self.adaptive = adaptive
        self.levels = levels
        self.inverse = inverse

        self.conv_params = dict(
            stride=(2,2),  #filters[0,0].shape,
            #  padding=tuple(np.array(filters[0,0].shape)//2),
            padding=tuple((np.array([lo.shape[0], hi.shape[0]])-1)//2),
            dilation=1,
            bias=None
        )

    @staticmethod
    def _get_1d_lo_and_hi(wavelet, inverse):
        if wavelet in pywt.wavelist(kind='discrete'):
            init_fn = None
            if inverse:
                lo, hi = pywt.Wavelet(wavelet).filter_bank[2:]
                # swap the filters.  Using the deconstruction filters for
                # reconstruction is important in a wavelet packet transform.
                hi = hi[::-1]
                lo = lo[::-1]
            else:
                lo, hi = pywt.Wavelet(wavelet).filter_bank[:2]
        else:  # assume it's a pytorch initialized vector like 'kaiming_normal:N' or 'normal:N'
            _init_fn_name, filter_size = wavelet.split(',', 1)
            filter_size = int(filter_size)
            lo, hi = np.zeros(filter_size, dtype='float32'), np.zeros(filter_size, dtype='float32')
            # lo and hi will be initialized later, so use zeros instead of
            # empty to ensure that initialization is called.
            init_fn = getattr(T.nn.init, _init_fn_name)
        lo, hi = T.tensor(lo, dtype=T.float), T.tensor(hi, dtype=T.float)
        return lo, hi, init_fn

    @staticmethod
    def compute_filters_2d(lo, hi):
        filters = T.stack([
            T.outer(lo, lo),
            T.outer(lo, hi),
            T.outer(hi, lo),
            T.outer(hi, hi)]).unsqueeze(1)  # shape: O, I=1, H_k, W_k
        return filters

    def get_max_level(self, input_shape:tuple[int]) -> int:
        H, W = input_shape[2:]
        s_h, s_w = self.conv_params['stride']
        J_max = min(  # solve for J the eqtn:  1 = H / stride**J
            math.ceil(np.log2(H) / np.log2(s_h)),
            math.ceil(np.log2(W) / np.log2(s_w)),)
        return J_max

    def forward(self, x:T.Tensor) -> T.Tensor:
        """
        Transform input images, doing either a Wavelet Packet transform or the
        inverse transform.
        Args:
            x: tensor of shape (B,C,H,W) or if inverse (B,C,L,H',W')
        Returns:
            Tensor of shape (B,C,D,H',W') or if inverse (B,C,H,W), where
              - D is the number of detail matrices, D=4**J for J wavelet levels
              - H' and W' are the size of the spatial dimension after transform
                H/2**(J-1) >= H' >= H/2**J  ... and similarly for W'
        """
        if self.adaptive == 1:
            filters = self.compute_filters_2d(self.lo, self.hi)
        else:
            assert self.adaptive in {0,2}, 'code bug: not implemented'
            filters = self.filters
        if self.inverse:
            return self._inverse_wavelet_packet_transform(x, filters)
        else:
            return self._wavelet_packet_transform(x, filters)

    def _ensure_even_num_rows_and_cols(self, tmp:T.Tensor):
        if tmp.shape[-2] % 2 == 1 or tmp.shape[-1] % 2 == 1:
            # ensure img has even number of rows and columns
            _pad_img = np.zeros((tmp.ndim*2, ), dtype='int32')
            if tmp.shape[-2] % 2 == 1: _pad_img[3] = 1
            if tmp.shape[-1] % 2 == 1: _pad_img[1] = 1
            tmp = T.nn.functional.pad(tmp, list(_pad_img))
        return tmp

    def _wavelet_packet_transform(self, x:T.Tensor, filters:T.Tensor) -> T.Tensor:
        B, I, H, W = x.shape
        J = self.levels
        if J == 'max':
            J = self.get_max_level(x.shape)
        elif J > (J_max := self.get_max_level(x.shape)):
            warnings.warn(
                f'The max wavelet levels for input data of shape {x.shape}'
                f' is {J_max}, but we have levels={J}.'
                '  Decrease levels or use smaller wavelet filters to avoid'
                ' wasting computation and ram usage.')
        tmp = x
        for lvl in range(J):
            #  padding: the convolution arithmetic with stride 2 says
            #      out = (inpt - filter_size + 1 + p) // 2
            #      ==> p = 2*out -1  - inpt + fsize
            #  since we want to assume 2*out == inpt, we have these constraints:
            #    - inpt is even (inpt%2 == 0)
            #    - p = fsize - 1 where p is the total num zeros to pad
            # compute the wavelet packet transform
            tmp = self._ensure_even_num_rows_and_cols(tmp)
            if lvl == 0:
                # assign the 4 filters to each input channel
                tmpfilters = filters.repeat(tmp.shape[1],1,1,1)
            else:
                tmpfilters = tmpfilters.repeat(4,1,1,1)
            #  print('forward a', tmp.shape)  # todo: padding issue
            tmp = T.conv2d(tmp, tmpfilters, groups=tmp.shape[1],
                           **self.conv_params)
            #  print('forward b', tmp.shape)  # todo: padding issue
        #  print('b', tmp.shape)
        return tmp.reshape(B, I, -1, *tmp.shape[-2:])

    def _inverse_wavelet_packet_transform(self, tmp:T.Tensor, filters:T.Tensor) -> T.Tensor:
        #  tmp = self._ensure_even_num_rows_and_cols(tmp)
        B, I, L, H, W = tmp.shape
        J = int(np.log2(L)/2)
        if not isinstance(self.levels, str):
            assert self.levels == J, 'sanity check input matches wavelet levels'
        # compute the inverse wavelet packet transform
        for lvl in range(J):
        #  print('a', tmp.shape)
            #  tmp = self._ensure_even_num_rows_and_cols(tmp)
            if lvl == 0:
                tmpfilters = filters.repeat(tmp.shape[1]*4**(J-1),1,1,1)
                assert tmpfilters.shape[0] == 4**J * tmp.shape[1], 'sanity check'
                tmp = tmp.reshape(B, I*L, H, W)
            else:
                #  z = tmpfilters[::4]  # undo the repeat op
                tmpfilters = filters.repeat(tmp.shape[1]//4,1,1,1)
            #  print('inverse a', tmp.shape)  # TODO: padding issue
            tmp = T.conv_transpose2d(
                tmp, tmpfilters, groups=tmp.shape[1], **self.conv_params)
            #  print('inverse b', tmp.shape)  # todo: padding issue
            # sum the filters
            assert L/4**lvl/4 == (L//4**lvl) // 4, 'sanity'
            assert I*L/4**(lvl+1) == int(I*L/4**(lvl+1)), 'sanity 2'
            tmp = tmp.reshape(B, I, (L//4**lvl) // 4, 4, *tmp.shape[2:])\
                    .sum(3)\
                    .reshape(B, int(I*L/4**(lvl+1)), *tmp.shape[2:])
        return tmp


if __name__ == "__main__":

    import skimage.data
    from deepfix import plotting as P
    from matplotlib import pyplot as plt
    plt.ion()
    #  im = T.tensor(skimage.data.cell() / 255.).unsqueeze(0).unsqueeze(0).float()
    im = T.tensor(skimage.data.camera() / 255.).unsqueeze(0).unsqueeze(0).float()
    #  im = T.tensor(np.random.randn(512, 512)).unsqueeze(0).unsqueeze(0).float()

    J = 3
    wp = WaveletPacket2d('coif1', J)
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

    # test parseval thm
    for J in [1,2,3,4,5,6,7,8]:
        wp = WaveletPacket2d('haar', J)
        assert T.allclose((wp(im)**2).sum(), (im**2).sum())

    # test max level
    wp = WaveletPacket2d('coif1', 'max')
    assert wp.get_max_level((None, None, 512, 512)) == 9
    print(wp(im).shape)
    assert wp(im).shape == (1, 1, 4**9, 1, 1)  # where coif1 has 6 filters, so h<6 and w<6
    wp = WaveletPacket2d('coif1', 8)
    print(wp(im).shape)
    assert wp(im).shape == (1, 1, 4**8, 2, 2)

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
    assert h == 512/2**J
    assert w == 512/2**J
    # test that wavelet packet doesn't mix channels
    wp = WaveletPacket2d('db1', 2)
    res3 = wp(imc)
    res3 = res3.reshape(1,-1,*res3.shape[-2:])
    P.plot_img_grid(res3[0])  # check that two images aren't mixed.

    # check that channels are properly separated from each other.  no mixing.
    wp = WaveletPacket2d('db1', 2)
    imd = T.stack([
        T.randn((100,100)),
        T.ones((100,100))]).unsqueeze(0)
    res4 = wp(imd)
    P.plot_img_grid(res4.reshape(-1,*res4.shape[-2:]))
    print(res4.shape)
    assert res4[0,0].abs().sum() > 1
    assert T.allclose(res4[0,1].abs().sum(), T.tensor(100**2/4))

    # test inverse wavelet packet transform is exact
    def test_inverse():
        #  im = T.tensor(skimage.data.camera() / 255.).unsqueeze(0).unsqueeze(0).float()
      for J in [2,3]:
        for wave in pywt.wavelist(kind='discrete'):
            if len(pywt.Wavelet(wave).dec_hi)*J >= 96//2:
                continue
                # ignore the large kernel sizes.  too much compute time.
                # note: tests should pass for these though

                #  if len(pywt.Wavelet(wave).dec_hi)*J >= 96:
                    #  im = T.tensor(np.random.randn(2*J*96, 2*J*96)).unsqueeze(0).unsqueeze(0).float()
                #  else:
                #  im = T.tensor(np.random.randn(2*J*96, 2*J*96)).unsqueeze(0).unsqueeze(0).float()
            else:
                im = T.tensor(np.random.randn(96, 96)).unsqueeze(0).unsqueeze(0).float()
            if wave == 'dmey': continue  # too large
            wp = WaveletPacket2d(wave, J)
            iwp = WaveletPacket2d(wave, J, inverse=True)
            print(f'testing {wave}')
            n = J*(len(pywt.Wavelet(wave).dec_hi)) # ignore boundary effects in the test.
            dist = ((iwp(wp(im))[..., n:-n,n:-n]  - im[..., n:-n,n:-n]).abs().max())
            assert T.allclose(iwp(wp(im))[..., n:-n,n:-n], im[..., n:-n,n:-n], atol=1e-5), f'{wave} {dist}'
    test_inverse()

    # plot inverse wavelet packet transform
    J = 3
    wp = WaveletPacket2d('coif1', J)
    iwp = WaveletPacket2d('coif1', J, inverse=True)
    res = wp(im)
    imrec = iwp(res)
    fig, (a1, a2) = plt.subplots(1,2)
    a1.imshow(im.squeeze())
    a2.imshow(imrec.squeeze())
    plt.show()

