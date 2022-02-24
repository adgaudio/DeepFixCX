import torch as T
import math
import warnings
from itertools import chain
#  import pytorch_colors as colors
#  import pywt
import pytorch_wavelets as pyw
from .wavelet_packet import WaveletPacket2d
from deepfix.models.api import get_resnet


class InvalidWaveletParametersError(Exception): pass


class DeepFixCompression(T.nn.Module):
    """Compress the input data via wavelet packet based feature extraction.
    No learning.
    """
    def __init__(self,
                 in_ch:int,
                 in_ch_multiplier:int,
                 # wavelet params
                 levels:int, wavelet:str,
                 # wavelet spatial feature extraction params
                 patch_size:int,
                 patch_features:list[str]=['sum_pos', 'sum_neg'],
                 how_to_error_if_input_too_small:str='raise',
                 ):
        """
        Args:
            in_ch:  Number of input channels
            in_ch_multiplier:  If >1, expand the input to in_ch*in_ch_multiplier
                channels using a random fixed 1x1 conv.
            levels:  Number of wavelet levels to use
            wavelet: The kind of wavelet.  Any discrete wavelet from pywt works.
            patch_size: The number of patches (in each dimension) for each
                coefficient matrix.  If patch_size=3, it makes 9 patches.
            patch_features: any of {'l1', 'l2', 'sum_pos', 'sum_neg'}.  The more
                choices, the less compressible the representation is.
                A reasonable choice is ['l1'] or ['l1'] or ['sum_pos', 'sum_neg'].
            how_to_error_if_input_too_small:  one of {'warn', 'raise'}.  The
                input image shape, wavelet levels, patch size (and also
                in_ch_multiplier and patch_features) should be chosen to ensure
                the model is actually doing compression.
        """
        super().__init__()
        self.how_to_error_if_input_too_small = how_to_error_if_input_too_small
        self.patch_features = patch_features
        #
        # construct the compressed encoder:
        #
        # preprocessing: expand channels by random projection
        assert in_ch_multiplier >= 1, 'invalid input. expect in_ch_multiplier >= 1'
        if in_ch_multiplier > 1:
            self.expand_input_channels = T.nn.Sequential(
                T.nn.Conv2d(in_ch, in_ch*in_ch_multiplier, 1),
                T.nn.CELU(),
            )
            # note: kaiming normal initialization of this step improves perf in xlm
            # This can be fixed randomly and not learned I think.
            T.nn.init.kaiming_normal_(self.expand_input_channels[0].weight, nonlinearity='sigmoid')
        else:
            self.expand_input_channels = T.nn.Identity()
        # wavelet transform
        self.wavelet_encoder = WaveletPacket2d(wavelet=wavelet, levels=levels)
        # wavelet feature extractor
        self.patch_size = patch_size
        # fix the model (no learning)
        for x in self.parameters():
            x.requires_grad_(False)
        # for convenience:  determine what the output shape should be
        if levels != 'max':
            D = self.get_n_extracted_features(
                J=levels, P=patch_size, F=len(self.patch_features))
        else:
            D = '?'
        self.out_shape = ("?", in_ch * in_ch_multiplier, D)

    def forward(self, x: T.Tensor):
        """Fixed weight dataset agnostic compression"""
        x = self.expand_input_channels(x)
        #
        # normalize the input so the output is more balanced.
        # not sure if this normalize step is helpful on color images.
        # it makes the extracted features have less extreme differences in value
        x = x - x.mean((-1, -2), keepdims=True)
        #
        x = self.spatial_feature_extractor(self.wavelet_encoder(x))
        return x

    def spatial_feature_extractor(self, data_2d: T.Tensor):
        # extract features from each spatial matrix
        B, C, D, h, w = data_2d.shape
        data_2d = data_2d.reshape(B, C*D, h, w)
        # normalize each coefficient matrix
        # TODO: decide whether to use
        #  data_2d /= data_2d.pow(2).sum((-1,-2), keepdims=True).sqrt()  # l2
        #  data_2d /= data_2d.abs().sum((-1,-2), keepdims=True)  # l1
        #
        _scores = []
        _zero = data_2d.new_tensor(0)
        p = self.patch_size
        if h<p or w<p:
            msg = (
                'Input data spatial dimensions are too small for choice'
                ' of wavelet level and patch size.  This is inefficient.'
                f' Patch size = {p}'
                f', level = {self.wavelet_encoder.levels}'
                f', shape_after_wavelet_transform={data_2d.shape}')
            if self.how_to_error_if_input_too_small == 'raise':
                raise InvalidWaveletParametersError(msg)
            elif self.how_to_error_if_input_too_small == 'warn':
                warnings.warn(msg)
            else:
                raise Exception(
                    f'unrecognized option, how_to_error_if_input_too_small={self.how_to_error_if_input_too_small}')
        #
        # reshape the (H,W) spatial data into a set of patches as needed.
        # zero pad rows and cols so can have a pxp grid
        # this isn't necessary if we have inputs with shape a power of 2 in each dimension.
        py, px = (p-h%p)%p, (p-w%p)%p  # total num cols and rows to pad
        assert py in set(range(p)) and px in set(range(p)), 'sanity check: code bug'
        yl = py//2
        yr = py-yl
        xl = px//2
        xr = px-xl
        lvl = T.nn.functional.pad(data_2d, (xl, xr, yl, yr))
        # ... unfold into patches, with spatial dimensions of patch last
        _,_,h,w = lvl.shape
        assert h%p == 0, w%p == 0
        lvl = T.nn.functional.unfold(
            lvl,
            kernel_size=(max(1,h//p), max(1,w//p)),
            stride=(max(1,h//p),max(1,w//p)))
        lvl = lvl.reshape(B,C*D,max(1,h//p),max(1,w//p),p*p)
        lvl = lvl.permute(0,1,4,2,3)  # put the spatial dimensions last.
        data_2d = lvl
        #
        # for each patch, get some numbers
        features = []
        if 'l1' in self.patch_features:
            #  features.append(data_2d.where(data_2d > 1e-6, _zero).abs().sum((-2, -1)).float())
            features.append(data_2d.abs().sum((-2, -1)).float())
        if 'sum' in self.patch_features:
            features.append(data_2d.sum((-2, -1)).float())
        if 'max' in self.patch_features:
            features.append(data_2d.max(-1).values.max(-1).values.float())
        if 'min' in self.patch_features:
            features.append(data_2d.min(-1).values.min(-1).values.float())
        if 'median' in self.patch_features:
            features.append(data_2d.view(*data_2d.shape[:-2], -1).median(-1).values.float())
        if 'l2' in self.patch_features:
            #  features.append(data_2d.where(data_2d > 1e-6, _zero).pow(2).sum((-2, -1)).float())
            features.append(data_2d.pow(2).sum((-2, -1)).float())
        if 'sum_pos' in self.patch_features:
            features.append(data_2d.where(data_2d > 1e-6, _zero).sum((-2, -1)).float())
        if 'sum_neg' in self.patch_features:
            features.append(data_2d.where(data_2d < -1e-6, _zero).sum((-2, -1)).float())
        _scores.append(T.stack(features, -1))
        out = T.cat([x.reshape(B,C,-1) for x in _scores], -1)
        if out[0].numel() > data_2d.numel():
            warnings.warn(
                f'{self.__class__.__name__}.patch_size too large, so the'
                '  compressed representation is larger than the input image.'
                '  Decrease the patch_size or wavelet levels.'
            )
        return out

    @staticmethod
    def get_n_extracted_features(J:int, P:int, F:int) -> int:
        """
        Args:
            J: number of wavelet levels
            P: patch size
            F: num features per patch
        Returns:
            Number of extracted features output by this compression encoder.
            The number is either exactly correct or an upper bound (See note).

        Note: The returned number is exact as long as the input data spatial
        dimensions are all larger than P, and it is an upper bound otherwise.
        For instance, if we have input data with a spatial shape (H,W)
        and H < p, then num_patches <= 1 * p.
        """
        d = 2  # num spatial dims is always 2
        num_patches = P**d
          # assume there are always p patches in each dimension
          # note: num_patches is correct as long as num pixels in each spatial
          # dimension is greater than p.  Otherwise, p**d is an upper bound.
        num_detail_matrices = 4**J
        num_features_per_patch = F
        return (num_detail_matrices * num_patches * num_features_per_patch)


class DeepFixCompression__OLD(T.nn.Module):
    """Compress the input data to ~1% of original size via feature extraction
    from a wavelet transform.  No learning.
    """
    def __init__(self,
                 in_ch:int,
                 in_ch_multiplier:int,
                 # wavelet params
                 levels:int, wavelet:str,
                 # wavelet spatial feature extraction params
                 patch_size:int,
                 ):
        super().__init__()
        #
        # construct the compressed encoder:
        #
        # preprocessing: expand channels by random projection
        assert in_ch_multiplier >= 1, 'invalid input. expect in_ch_multiplier >= 1'
        if in_ch_multiplier > 1:
            self.expand_input_channels = T.nn.Sequential(
                T.nn.Conv2d(in_ch, in_ch*in_ch_multiplier, 1),
                T.nn.CELU(),
            )
            # note: kaiming normal initialization of this step improves perf in xlm
            # This can be fixed randomly and not learned I think.
            T.nn.init.kaiming_normal_(self.expand_input_channels[0].weight, nonlinearity='sigmoid')
        else:
            self.expand_input_channels = T.nn.Identity()
        # wavelet transform
        self.wavelet_encoder = pyw.DWT(
            J=levels, wave=wavelet, mode='periodization')
        # wavelet feature extractor
        self.patch_size = patch_size
        # fix the model (no learning)
        for x in self.parameters():
            x.requires_grad_(False)
        # for convenience:  determine what the output shape should be
        D = self.get_n_wavelet_features(
            in_ch*in_ch_multiplier, wavelet_levels=levels, patch_size=patch_size)
        self.out_shape = ("?", in_ch * in_ch_multiplier, D)

    def forward(self, x: T.Tensor):
        """Fixed weight dataset agnostic compression"""
        x = self.expand_input_channels(x)
        x = self.wavelet_feature_extractor(*self.wavelet_encoder(x))
        return x

    def wavelet_feature_extractor(self, approx, detail):
        # extract features from each detail matrix
        _scores = []
        _zero = approx.new_tensor(0)
        p = self.patch_size
        for lvl_idx, lvl in enumerate(chain(detail, [approx.unsqueeze_(2)])):
            # divide the spatial dimensions into a pxp grid of "patches"
            # p=3 means we divide the detail matrix into 9 patches,
            # padding zeros as needed.
            h,w = lvl.shape[-2:]
            if p != 1 and (h>p or w>p):  # if condition avoids unnecessarily padding the input
                # zero pad rows and cols so can have a pxp grid
                py, px = (p-h%p)%p, (p-w%p)%p  # total num cols and rows to pad
                assert py in set(range(p)) and px in set(range(p)), 'sanity check: code bug'
                yl = py//2
                yr = py-yl
                xl = px//2
                xr = px-xl
                lvl = T.nn.functional.pad(lvl, (xl, xr, yl, yr))
                b,c,d,h,w = lvl.shape
                assert h%p == 0, w%p == 0
                if lvl_idx < len(detail):  # if not the approx matrix...
                    assert d == 3, 'sanity check: 3 detail matrices per level'
                lvl = T.nn.functional.unfold(
                    lvl.reshape(b,c*d,h,w),
                    kernel_size=(max(1,h//p), max(1,w//p)),
                    stride=(max(1,h//p),max(1,w//p)))
                lvl = lvl.reshape(b,c,d,max(1,h//p),max(1,w//p),p*p)
                # put the spatial dimensions last.
                lvl = lvl.permute(0,1,2,5,3,4)
            # for each patch, get some numbers
            _scores.append(T.stack([
                # pos and neg coefficients
                lvl.where(lvl > 1e-6, _zero).sum((-2, -1)).float(),
                lvl.where(lvl < -1e-6, _zero).sum((-2, -1)).float(),
                # the num of zeros and near zeros
                #  (lvl.abs() < 1e-6).sum((-2, -1)).float(),
                #  (lvl > 0).sum((-2,-1)).float(),
                #  (lvl < 0).sum((-2,-1)).float(),
                # num elements in this detail matrix (very redundant)
                #  T.ones(lvl.shape[:-2], device=approx.device, dtype=approx.dtype)
                   #  * lvl.shape[-2:].numel(),
            ], -1))
            del lvl
        # extract features comparing pairwise similarities over all detail matrices
        #  from simplepytorch.metrics import distance_correlation
        #  dcovs = [distance_correlation(coefs1.flatten(-2), coefs2.flatten(-2)).dcov
        #           for n,coefs1 in enumerate(detail)
        #           for m,coefs2 in enumerate(detail)
        #           if n != m]
        # flatten into a single vector per (image, channel)
        # the features of each (layer, detail matrix, patch)
        B, C = approx.shape[:2]
        out = T.cat([x.reshape(B,C,-1) for x in _scores], -1)
        if out[0].numel() > math.prod(detail[0].shape[-2:])*4:
            warnings.warn(
                f'{self.__class__.__name__}.patch_size too large, so the'
                '  compressed representation is larger than the input image.'
                '  Reduce the patch_size, or try reducing the wavelet levels.'
            )
        return out
                      #  approx.reshape(B, C, -1)], -1)

    @staticmethod
    def get_n_wavelet_features(in_channels, wavelet_levels, patch_size):
        # without 3x3 patches, much simpler
        #  n_detail_features = in_channels * wavelet_levels * 3*4  # 3 for horizontal,vertical,diagonal coefficients
        #  n_approx_features = in_channels*4
        #  n_dcov_features = 0  # in_channels * wavelet_levels * (wavelet_levels-1)
        #  return  n_detail_features+n_approx_features+n_dcov_features
        #
        # with pxp patches, more complex
        p = patch_size
        #  lvl_shape = [(math.ceil(H/2**i), math.ceil(W/2**i))
                    #  for i in range(1, wavelet_levels+1)]
        #  padded_lvl_shape = [(y + (p-y%p)%p, x + (p-x%p)%p) for y,x in lvl_shape]
        n_detail_features = [
            p*p  # num patches at this level
            * 3*2  # 3 detail matrices and 2 features per level.
            for _ in range(wavelet_levels)]

        # lvl_shapes = ...  # dynamic size
            #  p*p if (h>p or w>p) else h*w   # num patches at this level
            #  * 3*2  # 3 detail matrices and 2 features per level.
            #  for (h,w),_ in zip(lvl_shapes, range(wavelet_levels))]
        n_approx_features = n_detail_features[-1]/3
        rv = sum(n_detail_features) + n_approx_features
        assert rv == int(rv), 'sanity check: is integer value'
        return int(rv)


class DeepFixMLP(T.nn.Module):
    """Apply a multi-layer perceptron to the compressed DeepFix embedding space.
    The input to this module is the output of a DeepFixCompression(...) model.

    Expected input shape is (_, C, D), which corresponds to the output from
    DeepFixCompression, as defined by DeepFixCompression().out_shape
    """
    def __init__(self, C:int, D:int,
                 out_ch:int, depth:int, mid_ch:int, final_layer:T.nn.Module,
                 fix_weights='none'):
        """
        Args:
            C and D: Channels and dimension output by the DeepFixCompression model
            out_ch: Num outputs of the MLP.
            depth: Num hidden layers of MLP
            mid_ch: the size of the middle layers of MLP
            final_layer: A final activation fn to apply
            fix_weights: a str in {'none', 'all', 'all_except_fc'} useful to fix
                the weights so they are not part of backprop.  Used during
                testing to experiment with fixed weight networks, like extreme
                learning machines.
        """
        super().__init__()
        #  self.compression_encoder = compression_encoder
        self.spatial_attn = T.nn.Sequential(
            T.nn.Flatten(2),  # (B,C,D)
            VecAttn(D),  # (B,C,D)
            T.nn.Flatten(1)  # (B,C*D)
        )
        self.mlp = MLP(
            in_ch=C*D, out_ch=out_ch, depth=depth, mid_ch=mid_ch,
            final_activation_layer=final_layer)
        self.reset_parameters()
        if fix_weights == 'all':
            for x in self.parameters():
                x.requires_grad_(False)
        elif fix_weights == 'all_except_fc':
            params = list(self.lst.parameters())
            for x in params[:-1]:
                x.requires_grad_(False)
        else:
            assert fix_weights == 'none'

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B, C, D) where B is anything (batch size) and
                where C,D correspond to the channels and dimension output by a
                DeepFixCompression.
        """
        x = self.spatial_attn(x)
        x = self.mlp(x)
        return x

    @staticmethod
    def get_VecAttn_regularizer(mdl:T.nn.Module) -> T.Tensor:
        """Helper function to compute the l1 norm on the VecAttn weights.

        For each VecAttn layer in `mdl`, extract weights a vector,
        x = [x_1, ..., x_n], where `n` may vary for each VecAttn layer.
        Compute a normalized l1 norm for each VecAttnLayer:

            norm(x) = 1/N \sum_{i=1}^N |x_i|

        Then average the VecAttnLayers:

            z = mean(norm(x) for all VecAttn layers in model)

        The idea is to have an objective function like:

            Loss(model, dataset) + lbda * get_VecAttn_regularizer(model)

        where the get_VecAttn_regularizer(...) doesn't overpower the loss

        Args:
            mdl:  a pytorch module containing at least one VecAttn layer.

        Compute l1 norm || v_i ||_1 for each vector v_i of attention over the
        VecAttn modules.

        Returns: a scalar float value of type T.Tensor

        """
        #  return T.stack([
            #  T.linalg.norm(x.vec_attn.squeeze(), 1) for x in mdl.modules()
            #  if isinstance(x, VecAttn)]).mean()
        return T.stack([
            x.vec_attn.squeeze().abs().mean() for x in mdl.modules()
            if isinstance(x, VecAttn)]).mean()

    def reset_parameters(self):
        """Only modifies linear and conv layers!"""
        # I found that
        # using normal instead of uniform is important if the mlp is also fixed
        # (e.g. extreme learning machine style)
        # I guess uniform doesn't satisfy properties of random projections...
        # TODO: for different project: fan_out is better than fan_in on test
        # set IntelMobileODT when trying a completely fixed weight model (XLM),
        # but is it meaningful or just lucky?
        for l in self.modules():
            if isinstance(l, (T.nn.Linear, T.nn.modules.conv._ConvNd)):
                T.nn.init.kaiming_normal_(
                    l.weight.data, nonlinearity='sigmoid')  #, mode='fan_out')
                if l.bias is not None:
                    T.nn.init.normal_(
                        l.bias.data, mean=0, std=(2/l.weight.shape[0])**.5 / 10)


class DeepFixEnd2End(T.nn.Module):
    """When we can assume we have access to the full dataset,
    we can compress and use the MLP all in one step.
    This is useful for:
        - testing the compression method preserves discriminant information
        - experimenting with high resolution images
    """
    def __init__(self, compression: DeepFixCompression, mlp: DeepFixMLP):
        super().__init__()
        self.compression_mdl = compression
        self.mlp = mlp

    def forward(self, x):
        with T.no_grad():
            x = self.compression_mdl(x)
        x = self.mlp(x)
        return x


class Cosine(T.nn.Module):
    def forward(self, x):
        return x.cos()


class VecAttn(T.nn.Module):
    """Apply an attention weight to last dimension of given input.

    Given:
        input of shape (B,C,D)  with D features.
        attention of shape (1,1,D)  (a parameter)
    Perform:
        input * attention

        In linear algebra:  (I A) where
            input I is of shape (B,C,D)
            and weights A = diag(attention_weights) is (D,D)

    This is useful to give weight to each dimension in D.  For each feature d_i
    in D, we have a (B,C) sub-matrix.  This attention weight determines the
    relative importance of that feature d_i.
    """
    def __init__(self, D):
        super().__init__()
        self.vec_attn = T.nn.Parameter(T.rand((1,1,D)))

    def forward(self, x):
        """
        Args:
            x: tensor of shape (B,C,D) where B and C don't matter,
                and D represents the D features we compute attention over.
        """
        _shape_should_not_change = x.shape
        ret = x * self.vec_attn
        assert _shape_should_not_change == ret.shape, 'code error'
        return ret


class MLP(T.nn.Module):
    """Multi-Layer Perceptron with CELU activations"""
    def __init__(self, in_ch, out_ch, depth=8, mid_ch=None,
                 final_activation_layer:T.nn.Module=None,):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        lst = []
        lst.append(T.nn.Sequential(
            T.nn.Flatten(), T.nn.Linear(in_ch, mid_ch, bias=True),T.nn.BatchNorm1d(mid_ch),T.nn.CELU()))
        # add linear -> celu layers
        for _ in range(depth-1):
            lst.extend(T.nn.Sequential(T.nn.Linear(mid_ch, mid_ch, bias=True),T.nn.BatchNorm1d(mid_ch),T.nn.CELU()))
        self.features = T.nn.Sequential(*lst)
        if final_activation_layer is None:
            self.fc = T.nn.Sequential(T.nn.Linear(mid_ch, out_ch, bias=True))
        else:
            self.fc = T.nn.Sequential(T.nn.Linear(mid_ch, out_ch, bias=True),
                                      final_activation_layer)
#LayerNorm
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


#  class ConvertColorspace(T.nn.Module):
    #  def __init__(self, fn=colors.rgb_to_lab):
        #  super().__init__()
        #  self.fn = fn
    #  def forward(self, x):
        #  return self.fn(x)


def get_DeepFixEnd2End(
        in_channels, out_channels,
        in_ch_multiplier=1, wavelet='coif1', wavelet_levels=4, wavelet_patch_size=1,
        mlp_depth=2 , mlp_channels=None, mlp_activation=None,
        mlp_fix_weights='none', patch_features='l1'):
    enc = DeepFixCompression(
        in_ch=in_channels, in_ch_multiplier=in_ch_multiplier, levels=wavelet_levels,
        wavelet=wavelet, patch_size=wavelet_patch_size, patch_features=patch_features.split(','))
    C, D = enc.out_shape[-2:]
    mlp = DeepFixMLP(
        C=C, D=D, out_ch=out_channels, depth=mlp_depth, mid_ch=mlp_channels,
        final_layer=mlp_activation, fix_weights=mlp_fix_weights)
    m = DeepFixEnd2End(enc, mlp)
    return m


class DeepFixClassifier(T.nn.Module):
    def __init__(self, backbone:str, backbone_pretraining:str, in_channels:int, out_channels:int, patch_size:int):
        super().__init__()

        # figure out how many layers gets us to a 64x64 img via upsampling
        #  by adding 2 rows and 2 cols each time.
        # also ensure the total number of elements is about the same so the ram
        # usage stays small
        num_layers = math.ceil(max((64 - patch_size) / 2, 0))
        lst, _out_chan = [], in_channels
        for l in range(1, 1+num_layers):
            num_pixels = (patch_size + 2*l)**2
            _in_chan, _out_chan = _out_chan, math.ceil(in_channels*patch_size**2 / num_pixels)
            lst.append(T.nn.Sequential(
                T.nn.ConvTranspose2d(_in_chan, _out_chan, 3), T.nn.SELU()))
        self.upsampler = T.nn.Sequential(*lst)
        mid_channels = _out_chan

        if backbone.startswith('resnet'):
            self.backbone = get_resnet(
                backbone, backbone_pretraining, mid_channels, out_channels)
        else:
            raise NotImplementedError(f'backbone={backbone}')
        #      *[T.nn.Sequential(
        #          T.nn.ConvTranspose2d(in_channels, in_channels, 3), T.nn.SELU())
        #      for _ in range(num_layers)])
        self.in_channels = in_channels
        self.patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.in_channels, self.patch_size, self.patch_size)
        x = self.upsampler(x)
        x = self.backbone(x)
        return x


def get_DeepFixEnd2End_v2(
        in_channels, out_channels,
        in_ch_multiplier=1, wavelet='coif1', wavelet_levels=4, wavelet_patch_size=1,
        patch_features='l1', backbone='resnet18', backbone_pretraining='imagenet'):
    enc = DeepFixCompression(
        in_ch=in_channels, in_ch_multiplier=in_ch_multiplier, levels=wavelet_levels,
        wavelet=wavelet, patch_size=wavelet_patch_size, patch_features=patch_features.split(','))
    enc_channels = 4**wavelet_levels*in_channels*in_ch_multiplier*len(patch_features.split(','))

    classifier = DeepFixClassifier(
        backbone=backbone, backbone_pretraining=backbone_pretraining,
        in_channels=enc_channels, out_channels=out_channels,
        patch_size=wavelet_patch_size)

    m = DeepFixEnd2End(enc, classifier)
    return m


if __name__ == "__main__":
    m = get_DeepFixEnd2End(3, out_channels=30, )
