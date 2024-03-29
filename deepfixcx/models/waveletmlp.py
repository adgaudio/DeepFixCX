import torch as T
import torchvision.transforms as tvt
from typing import List, Tuple, Optional, Union
import math
import warnings
from itertools import chain
#  import pytorch_colors as colors
#  import pywt
# import pytorch_wavelets as pyw
from .wavelet_packet import WaveletPacket2d
from deepfixcx.models.api import get_resnet


class InvalidWaveletParametersError(Exception): pass


def astuple2(x:Union[int,Tuple[int,int]]):
    if isinstance(x, tuple):
        return x
    return (x,x)

class DeepFixCXCompression(T.nn.Module):
    """Compress the input data via wavelet packet based feature extraction.
    No learning.
    """
    def __init__(self,
                 in_ch:int,
                 in_ch_multiplier:int,
                 # wavelet params
                 levels:int, wavelet:str,
                 # wavelet spatial feature extraction params
                 patch_size:Union[int,Tuple[int,int]],
                 patch_features:list[str]=['l1'],
                 how_to_error_if_input_too_small:str='raise',
                 zero_mean:bool=False,
                 adaptive:int=0
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
            how_to_error_if_input_too_small: one of {'warn', 'raise', 'ignore'}.
                The input image shape, wavelet levels, patch size (and also
                in_ch_multiplier and patch_features) should be chosen to ensure
                the model is actually doing compression.
            zero_mean:  If True, ensure each input channel has zero mean before
                wavelet transform.
        """
        super().__init__()
        self.how_to_error_if_input_too_small = how_to_error_if_input_too_small
        assert isinstance(patch_features, (list, tuple))
        self.patch_features = patch_features
        self.zero_mean = zero_mean
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
        self.wavelet_encoder = WaveletPacket2d(
            wavelet=wavelet, levels=levels, adaptive=adaptive)
        # wavelet feature extractor
        self.patch_size = astuple2(patch_size)
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
        if self.zero_mean:
            x = x - x.mean((-1, -2), keepdims=True)
        #
        x = self.spatial_feature_extractor(self.wavelet_encoder(x))
        return x

    def spatial_feature_extractor(self, data_2d: T.Tensor):
        # extract features from each spatial matrix
        B, C, D, h, w = data_2d.shape
        data_2d = data_2d.reshape(B, C*D, h, w)
        # normalize each coefficient matrix
        #  data_2d = data_2d.log2() / T.log2(data_2d.new_tensor(4))  # log base 4
        #  data_2d = data_2d.log2() # log base 2
        # TODO: decide whether to use
        #  data_2d /= data_2d.pow(2).sum((-1,-2), keepdims=True).sqrt()  # l2
        #  data_2d /= data_2d.abs().sum((-1,-2), keepdims=True)  # l1
        #
        _scores = []
        _zero = data_2d.new_tensor(0)
        ph, pw = self.patch_size
        if h<ph or w<pw:
            msg = (
                'Input data spatial dimensions are too small for choice'
                ' of wavelet level and patch size.  This is inefficient.'
                f' Patch size = {(ph, pw)}'
                f', level = {self.wavelet_encoder.levels}'
                f', shape_after_wavelet_transform={data_2d.shape}')
            if self.how_to_error_if_input_too_small == 'raise':
                raise InvalidWaveletParametersError(msg)
            elif self.how_to_error_if_input_too_small == 'warn':
                warnings.warn(msg)
            elif self.how_to_error_if_input_too_small == 'ignore':
                pass
            else:
                raise Exception(
                    f'unrecognized option, how_to_error_if_input_too_small={self.how_to_error_if_input_too_small}')
        #
        # reshape the (H,W) spatial data into a set of patches as needed.
        # zero pad rows and cols so can have a pxp grid
        # this isn't necessary if we have inputs with shape a power of 2 in each dimension.
        py, px = (ph-h%ph)%ph, (pw-w%pw)%pw  # total num cols and rows to pad
        assert py in set(range(ph)) and px in set(range(pw)), 'sanity check: code bug'
        yl = py//2
        yr = py-yl
        xl = px//2
        xr = px-xl
        lvl = T.nn.functional.pad(data_2d, (xl, xr, yl, yr))
        # ... unfold into patches, with spatial dimensions of patch last
        _,_,h,w = lvl.shape
        assert h%ph == 0, w%pw == 0
        lvl = T.nn.functional.unfold(
            lvl,
            kernel_size=(max(1,h//ph), max(1,w//pw)),
            stride=(max(1,h//ph),max(1,w//pw)))
        lvl = lvl.reshape(B,C*D,max(1,h//ph),max(1,w//pw),ph*pw)
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
    def get_n_extracted_features(J:int, P:Union[int,Tuple[int,int]], F:int) -> int:
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
        Ph, Pw = astuple2(P)
        num_patches = Ph * Pw
          # assume there are always p patches in each dimension
          # note: num_patches is correct as long as num pixels in each spatial
          # dimension is greater than p.  Otherwise, p**d is an upper bound.
        num_detail_matrices = 4**J
        num_features_per_patch = F
        return (num_detail_matrices * num_patches * num_features_per_patch)

    @staticmethod
    def reconstruct(deepfixcx_embedding: T.Tensor, orig_img_HW:Tuple[int,int],
                    wavelet:str, J:int, P:Union[int,Tuple[int,int]],
                    restore_orig_size:bool=True,
                    min_size:Optional[Tuple[int,int]]=None) -> T.Tensor:
        """Use the inverse wavelet transform to reconstruct a deepfixcx embedding.
        This assumes patch_features is only one feature, like "l1" or "sum".
        It "unpools" the patches by repeating the value of each patch.
        The pooling function may cause output values outside of [0,1].  You
        could normalize the output values into [0,1] by clipping them with
        `tensor.clamp(0,1)`, or do nothing.
        Args:
            deepfixcx_embedding:  The output of DeepFixCXCompression.forward(...), of shape (B, ...)
            orig_img_HW:  a tuple like (H, W) denoting spatial height and spatial width of original input image.
            wavelet: the value passed to DeepFixCXCompression
            J: the wavelet level passed to DeepFixCXCompression
            P: the patch size passed to DeepFixCXCompression
            restore_orig_size: If True, reconstruct to the original input size by
                unpooling.  Otherwise, reconstruct to some smaller size.
            min_size: If supplied, ensure the height and width dimension are each as
                least as large as given (H,W) tuple.
        Returns:
            Image of shape (B, C, H, W) corresponding to reconstruction of
            original input image.
        """
        fn = DeepFixCXReconstruct(
            orig_img_HW=orig_img_HW, wavelet=wavelet, J=J, P=P,
            restore_orig_size=restore_orig_size).to(
                deepfixcx_embedding.device, deepfixcx_embedding.dtype)
        return fn(deepfixcx_embedding)


class DeepFixCXReconstruct(T.nn.Module):
    """Use the inverse wavelet transform to reconstruct a deepfixcx embedding.
    This assumes patch_features is only one feature, like "l1" or "sum".
    It "unpools" the patches by repeating the value of each patch.
    The pooling function may cause output values outside of [0,1].  You
    could normalize the output values into [0,1] by clipping them with
    `tensor.clamp(0,1)`, or do nothing.
    Args:
        deepfixcx_embedding:  The output of DeepFixCXCompression.forward(...), of shape (B, ...)
        orig_img_HW:  a tuple like (H, W) denoting spatial height and spatial width of original input image.
        wavelet: the value passed to DeepFixCXCompression
        J: the wavelet level passed to DeepFixCXCompression
        P: the patch size passed to DeepFixCXCompression
        restore_orig_size: If True, reconstruct to the original input size by
            unpooling.  Otherwise, reconstruct to some smaller size.
        min_size: If supplied, ensure the height and width dimension are each as
            least as large as given (H,W) tuple.
    Returns:
        Image of shape (B, C, H, W) corresponding to reconstruction of
        original input image.
    """
    def __init__(self, wavelet:str, J:int, P:Union[int, Tuple[int,int]],
                 restore_orig_size:bool=True,
                 min_size:Optional[Tuple[int,int]]=None,
                 orig_img_HW:Tuple[int,int]=(None, None),
                 ):
        super().__init__()
        self.restore_orig_size = restore_orig_size
        self.wavelet, self.J, self.P = wavelet, J, astuple2(P)
        self.iwp = WaveletPacket2d(levels=J,wavelet=wavelet,inverse=True)
        self.H, self.W = orig_img_HW
        self.min_size = min_size

    def forward(self, deepfixcx_embedding:T.Tensor,
                orig_img_HW:Optional[Tuple[int,int]]=None):
        J, (Ph, Pw) = self.J, self.P
        if orig_img_HW is None:
            H, W = self.H, self.W
        else:
            H, W = orig_img_HW
        assert H is not None
        assert W is not None
        B = deepfixcx_embedding.shape[0]
        repY, repX = int(math.ceil(H/2**J/Ph)), int(math.ceil(W/2**J/Pw))
        deepfixcx_embedding = deepfixcx_embedding.reshape(B,-1,4**J,Ph,Pw)
        # unpool
        if self.restore_orig_size:
            deepfixcx_embedding = deepfixcx_embedding\
                    .repeat_interleave(repX, dim=-1)\
                    .repeat_interleave(repY, dim=-2)
        # normalize
        deepfixcx_embedding = deepfixcx_embedding / (repY*repX)
        # get the reconstruction
        recons = self.iwp(deepfixcx_embedding)
        del deepfixcx_embedding
        if self.restore_orig_size:
            # ... restore original size by removing any padding created by deepfixcx
            recons = tvt.CenterCrop((H, W))(recons)
        if self.min_size is not None:
            h = max(self.min_size[0], recons.shape[-2])
            w = max(self.min_size[1], recons.shape[-1])
            recons = tvt.Resize((h,w))(recons)
        return recons


class Normalization(T.nn.Module):
    """Expect input of shape  (B,C,D) and normalize each of the D features"""
    def __init__(self, D:int, normalization:str, filepath:str=None):
        super().__init__()
        if normalization == 'batchnorm':
            self.normfn = T.nn.BatchNorm1d(D)
        elif normalization == 'none':
            self.normfn = T.nn.Identity()
        elif normalization in {'whiten', '0mean'}:
            dct = T.load(filepath)
            mu, var = dct['means'], dct['vars']
            mu, var = mu.reshape(1,1,D), var.reshape(1,1,D)
            std = var.sqrt()
            self.register_buffer('mu', mu)
            self.register_buffer('std', std)
            if normalization == 'whiten':
                self.normfn = self.whiten
            elif normalization == '0mean':
                self.normfn = self.zero_mean
            else:
                raise NotImplementedError('code bug')
        else:
            raise NotImplementedError(f'Unrecognized normalization: {normalization}')

    def whiten(self, x):
        return (x - self.mu) / self.std

    def zero_mean(self, x):
        return (x - self.mu)

    def forward(self, x):
        """Expect input of shape  (B,C,D) and normalize the features on dimension D"""
        #  print(1, x.mean(0).mean().item())
        x = self.normfn(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        #  print(2, x.mean(0).mean().item())
        return x


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


class SoftmaxVecAttn(T.nn.Module):
    """For testing.  Use LogSoftmaxVecAttn instead"""
    def __init__(self, D):
        super().__init__()
        self.vec_attn = T.nn.Parameter(T.rand((1,1,D)))

    def forward(self, x):
        return x * self.vec_attn.softmax(-1)


class LogSoftmaxVecAttn(T.nn.Module):
    """Apply an attention weight to last dimension of given input.

    Given:
        input of shape (B,C,D)  with D features.
        attention of shape (1,1,D)  (a parameter)
    Perform:
        input * (attention.log_softmax(-1))
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
        ret = x * self.vec_attn.log_softmax(-1)
        assert _shape_should_not_change == ret.shape, 'code error'
        return ret


class DeepFixCXMLP(T.nn.Module):
    """Apply a multi-layer perceptron to the compressed DeepFixCX embedding space.
    The input to this module is the output of a DeepFixCXCompression(...) model.

    Expected input shape is (_, C, D), which corresponds to the output from
    DeepFixCXCompression, as defined by DeepFixCXCompression().out_shape
    """
    def __init__(self, C:int, D:int,
                 out_ch:int, depth:int, mid_ch:int, final_layer:T.nn.Module,
                 fix_weights:str='none', input_normalization:List='none',
                 attn_class:T.nn.Module=VecAttn):
        """
        Args:
            C and D: Channels and dimension output by the DeepFixCXCompression model
            out_ch: Num outputs of the MLP.
            depth: Num hidden layers of MLP
            mid_ch: the size of the middle layers of MLP
            final_layer: A final activation fn to apply
            fix_weights: a str in {'none', 'all', 'all_except_fc'} useful to fix
                the weights so they are not part of backprop.  Used during
                testing to experiment with fixed weight networks, like extreme
                learning machines.
            input_normalization:  a list of args that tells Normalization class how
                to normalize each of the D features. Any of these:  {
                ['none'], ['batchnorm'], ['file', filepath]}
            attn_class:  Either VecAttn or SoftmaxVecAttn (or a suitable T.nn.Module)
        """
        super().__init__()
        #  self.compression_encoder = compression_encoder
        self.spatial_attn = T.nn.Sequential(
            T.nn.Flatten(2),  # (B,C,D)
            Normalization(D, *input_normalization),
            attn_class(D),  # (B,C,D)
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
                DeepFixCXCompression.
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


class DeepFixCXEnd2End(T.nn.Module):
    """When we can assume we have access to the full dataset,
    we can compress and use the MLP all in one step.
    This is useful for:
        - testing the compression method preserves discriminant information
        - experimenting with high resolution images
    """
    def __init__(self, compression: DeepFixCXCompression, mlp: DeepFixCXMLP):
        super().__init__()
        self.compression_mdl = compression
        self.mlp = mlp

    def forward(self, x):
        x = self.compression_mdl(x)
        x = self.mlp(x)
        return x


class Cosine(T.nn.Module):
    def forward(self, x):
        return x.cos()


class MLP(T.nn.Module):
    """Multi-Layer Perceptron with CELU activations"""
    def __init__(self, in_ch, out_ch, depth=8, mid_ch=None,
                 final_activation_layer:T.nn.Module=None,):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        lst = []
        lst.append(T.nn.Sequential(
            T.nn.Flatten(), T.nn.Linear(in_ch, mid_ch, bias=True), T.nn.CELU()))
        # add linear -> celu layers
        for _ in range(depth-1):
            lst.extend(T.nn.Sequential(T.nn.Linear(mid_ch, mid_ch, bias=True), T.nn.CELU()))
        self.features = T.nn.Sequential(*lst)
        if final_activation_layer is None:
            self.fc = T.nn.Sequential(T.nn.Linear(mid_ch, out_ch, bias=True))
        else:
            self.fc = T.nn.Sequential(T.nn.Linear(mid_ch, out_ch, bias=True),
                                      final_activation_layer)

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


class DeepFixCXImg2Img(T.nn.Module):
    """Privatize and compress an Image, outputting another image.  Useful as a
    layer before a deep network.  No learning happens in this layer by default.
    """
    def __init__(self, in_channels, J:int, P:Union[int,Tuple[int,int]], wavelet='db1', patch_features='l1',
                 restore_orig_size:bool=False, min_size:Optional[Tuple[int,int]]=None,
                 how_to_error_if_input_too_small='raise'):
        super().__init__()
        self.enc = DeepFixCXCompression(
            in_ch=in_channels, in_ch_multiplier=1,
            # wavelet params
            levels=J, wavelet=wavelet,
            # wavelet spatial feature extraction params
            patch_size=P, patch_features=patch_features.split(','),
            zero_mean=False, adaptive=0,
            how_to_error_if_input_too_small=how_to_error_if_input_too_small
        )
        self.recon = DeepFixCXReconstruct(
            wavelet=wavelet, J=J, P=P,
            restore_orig_size=restore_orig_size, min_size=min_size)

    def forward(self, x:T.Tensor):
        orig_img_HW = x.shape[-2:]
        x = self.enc(x)
        x = self.recon(x, orig_img_HW=orig_img_HW)
        #  print(x.shape)
        #  from matplotlib import pyplot as plt
        #  plt.figure(0)
        #  plt.imshow(x[0].permute(1,2,0).cpu().numpy())
        #  plt.show(block=False)
        #  plt.pause(.001)
        return x


def get_DeepFixCXEnd2End(
        in_channels, out_channels,
        in_ch_multiplier=1, wavelet='coif2', wavelet_levels=4, wavelet_patch_size=1,
        mlp_depth=2 , mlp_channels=None, mlp_activation=None,
        mlp_fix_weights='none', patch_features='l1',
        zero_mean:bool=False, normalization=('none', ), mlp_attn='LogSoftmaxVecAttn',
        adaptive:int=0):
    enc = DeepFixCXCompression(
        in_ch=in_channels, in_ch_multiplier=in_ch_multiplier, levels=wavelet_levels,
        wavelet=wavelet,
        patch_size=wavelet_patch_size, patch_features=patch_features.split(','),
        zero_mean=zero_mean, adaptive=adaptive)
    C, D = enc.out_shape[-2:]
    mlp = DeepFixCXMLP(
        C=C, D=D, out_ch=out_channels, depth=mlp_depth, mid_ch=mlp_channels,
        final_layer=mlp_activation, fix_weights=mlp_fix_weights,
        input_normalization=normalization,
        attn_class={'Identity': T.nn.Identity,
                    'VecAttn': VecAttn,
                    'SoftmaxVecAttn': SoftmaxVecAttn,
                    'LogSoftmaxVecAttn': LogSoftmaxVecAttn,
                    }[mlp_attn],
    )
    m = DeepFixCXEnd2End(enc, mlp)
    return m


class DeepFixCXClassifier(T.nn.Module):
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


def get_DeepFixCXEnd2End_v2(
        in_channels, out_channels,
        in_ch_multiplier=1, wavelet='coif2', wavelet_levels=4, wavelet_patch_size=1,
        patch_features='l1', backbone='resnet18', backbone_pretraining='imagenet'):
    enc = DeepFixCXCompression(
        in_ch=in_channels, in_ch_multiplier=in_ch_multiplier, levels=wavelet_levels,
        wavelet=wavelet, patch_size=wavelet_patch_size, patch_features=patch_features.split(','))
    enc_channels = 4**wavelet_levels*in_channels*in_ch_multiplier*len(patch_features.split(','))

    classifier = DeepFixCXClassifier(
        backbone=backbone, backbone_pretraining=backbone_pretraining,
        in_channels=enc_channels, out_channels=out_channels,
        patch_size=wavelet_patch_size)

    m = DeepFixCXEnd2End(enc, classifier)
    return m


if __name__ == "__main__":
    m = get_DeepFixCXEnd2End(3, out_channels=30, )
