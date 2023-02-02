from typing import Optional, Tuple, Union, List
from torch.utils.data import DataLoader, Subset, Dataset
from dataclasses import dataclass
import torch.utils.data
import os
import numpy as np
import torchvision.transforms as tvt
import torchvision.datasets as tvd
from sklearn.model_selection import StratifiedShuffleSplit
import timm.data.loader

import torch.nn
from simplepytorch import datasets as D
from deepfixcx.models import DeepFixCXImg2Img


@dataclass
class KimEyeArgs:
    """Experiment with a different way to define and parse arguments to a
    python function"""
    train_frac: float
    test_frac: float
    random_state: int = np.random.default_rng().integers(1<<32)
    batch_size: int = int(os.environ.get('batch_size', 20))

    def __post_init__(self):
        # parse to correct type
        for k, v in self.__dataclass_fields__.items():
            if v.type in {float, str, int}:
                val = getattr(self, k)
                if isinstance(val, str):
                    setattr(self, k, v.type(val))


def get_dset_kimeye(*args, **kwargs):
    args = KimEyeArgs(*args, **kwargs)
    dset = D.PreProcess(D.KimEye(), lambda dct: (
            tvt.Compose([
                tvt.ToTensor(), tvt.CenterCrop(240), ])(dct['image']),
            torch.tensor(dct['label']),
    ))

    N = len(dset)
    # get a train and test set
    idxs_train, idxs_test = list(StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_frac, random_state=args.random_state).split(
            np.arange(N), dset.labels))[0]
    # split train into train/val
    if args.train_frac + args.test_frac < 1:
        val_frac = 1 - args.train_frac - args.test_frac
        idxs_train, idxs_val = list(StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac / (1-args.test_frac),
            random_state=args.random_state+1).split(
                idxs_train, dset.labels[idxs_train]))[0]
    else:
        idxs_val = None
    # datasets
    dct = {
        'train_dset': Subset(dset, idxs_train),
        'val_dset': Subset(dset, idxs_val) if idxs_val is not None else None,
        'test_dset': Subset(dset, idxs_test),
    }
    # print({k: len(v) for k, v in dct.items()}, args.random_state)
    # dataloaders
    dct.update({
        'train_loader': DataLoader(
            dct['train_dset'], batch_size=args.batch_size, shuffle=True),
        'val_loader': DataLoader(
            dct['val_dset'], batch_size=args.batch_size, shuffle=False) if dct['val_dset'] is not None else None,
        'test_loader': DataLoader(
            dct['test_dset'], batch_size=args.batch_size, shuffle=False),
    })
    return dct, D.KimEye.CLASS_NAMES


class RandomSampler(torch.utils.data.Sampler):
    """Randomly sample without replacement a subset of N samples from a dataset
    of M>N samples.  After a while, a new subset of N samples is requested but
    (nearly) all M dataset samples have been used.  When this happens, reset
    the sampler, but ensure that N samples are returned.
    (In this case, the same image may appear in the that batch twice).

    This is useful to create smaller epochs, where each epoch represents only N
    randomly chosen images of a dataset of M>N images, and where random
    sampling is without replacement.

    """
    def __init__(self, data_source, num_samples:int):
        """
        Args:
            data_source:  pytorch Dataset class or object with __len__
                          ****Assume len(data_source) does not change.****
        """
        super().__init__(data_source)
        assert num_samples > 0
        assert num_samples <= len(data_source)
        self.dset_length = len(data_source)
        self.num_samples = num_samples
        self._cur_idx = 0
        self._ordering = self.new_ordering()

    def new_ordering(self):
        return torch.randperm(self.dset_length, dtype=torch.long)

    def next_idxs(self):
        if self._cur_idx + self.num_samples >= len(self._ordering):
            some_idxs = self._ordering[self._cur_idx:self._cur_idx+self.num_samples]
            self._ordering = self.new_ordering()
            self._cur_idx = self.num_samples-len(some_idxs)
            more_idxs = self._ordering[0:self._cur_idx]
            #  print(some_idxs, more_idxs)
            idxs = torch.cat([some_idxs, more_idxs])
        else:
            idxs = self._ordering[self._cur_idx:self._cur_idx+self.num_samples]
            self._cur_idx += self.num_samples
        return idxs.tolist()

    def __iter__(self):
        yield from self.next_idxs()

    def __len__(self):
        return self.num_samples


def onehot(y, nclasses):
    return torch.zeros((y.numel(), nclasses), dtype=y.dtype, device=y.device)\
            .scatter_(1, y.unsqueeze(1), 1)


class ResizeCenterCropTo(torch.nn.Module):
    """Resize tensor image to desired shape yx=(y, x) by enlarging the image
    without changing its aspect ratio, then center cropping.

    This can cut out the boundaries of images.
    """
    def __init__(self, yx:Tuple[int,int]):
        super().__init__()
        self.yx = yx
        self.center_crop = tvt.CenterCrop(yx)

    def resize_preserve_aspect(self, x:torch.Tensor):
        """
        Args:
            x: Tensor image of shape (?, H, W) or other shape accepted by
                tvt.functional.resize"""
        given_shape = x.shape[-2:]
        sufficiently_large = np.array(given_shape) >= self.yx
        if np.all(sufficiently_large):
            return x
        desired_aspect = self.yx[0] / self.yx[1]
        given_aspect = given_shape[0] / given_shape[1]
        if given_aspect < desired_aspect:
            # resize the y dim based on x
            resize_to = (self.yx[1], round(self.yx[0] / given_shape[0] * given_shape[1]))
        elif given_aspect > desired_aspect:
            # resize the x dim based on y
            resize_to = (round(self.yx[1] / given_shape[1] * given_shape[0]), self.yx[0])
        else:
            # same aspect ratio. just resize the image to yx
            resize_to = self.yx
        x = tvt.functional.resize(x, resize_to)
        #  print(x.shape)
        return x

    def forward(self, x):
        x = self.resize_preserve_aspect(x)
        x = self.center_crop(x)
        return x


def group_random_split(
        desired_split_fracs:list[float], group:np.ndarray, rng=None):
    # in code below, imagine we have images, each belonging to a patient
    # and we want to ensure no patient is mixed across splits.
    assert all(0 <= x <= 1 for x in desired_split_fracs), 'desired_split_fracs must contain values in (0,1)'
    assert sum(desired_split_fracs) <= 1.5, 'should satisfy sum(desired_split_fracs) <= 1+eps, with some margin for error'
    # count the patients
    patients, lookup_patient_idx, counts = np.unique(
        group, return_inverse=True, return_counts=True)
    # sanity check: if any desired split is smaller than the size of a single patient,
    # there may be sampling problems where some splits are empty.
    assert min([x for x in desired_split_fracs if x != 0]) >= np.max(counts / len(group)), f'minimum allowed split fraction is >= {np.max(counts) / len(group)}'
    # randomly shuffle the patients to get an ordering over them
    if rng is None:
        rng = np.random.default_rng()
    idxs = rng.permutation(np.arange(len(patients)))
    # compute what fraction of total images we get by considering the first N
    # patients for all N.
    fracs = np.cumsum(counts[idxs]) / len(group)
    # split the data, ensuring the patients have equal chance of appearing in either set.
    img_idxs = np.arange(len(group))
    assert len(img_idxs) == len(lookup_patient_idx), 'code bug'
    splits = []
    _n_patients_so_far = 0
    _frac_so_far = 0
    for desired_frac in desired_split_fracs:
        if desired_frac == 0:
            splits.append(np.array([]))
        else:
            # define some "cut point" / threshold at which we reach the desired_frac
            n_patients = np.digitize(desired_frac+_frac_so_far, fracs, False)
            #  print(fracs, desired_frac, _frac_so_far, n_patients, n_patients - _n_patients_so_far)
            # get the indices of the samples that correspond to these patients
            splits.append(img_idxs[np.isin(lookup_patient_idx, idxs[_n_patients_so_far:n_patients])])
            # bookkeeping
            _n_patients_so_far = n_patients
            _frac_so_far = fracs[n_patients-1]
            #  _frac_so_far += desired_frac
    return splits


def get_dset_chexpert(train_frac=.8, val_frac=.2, small=False,
                      labels:str='diagnostic', num_identities=None,
                      epoch_size:Optional[int]=None
                      ):
    """
    Args:
        labels:  either "diagnostic" (the 14 classes defined as
            D.CheXpert.LABELS_DIAGNOSTIC) or "identity" ("patient", "study",
            "view", "index").
        small:  whether to use CheXpert_Small dataset (previously downsampled
            images) or the fully size dataset.
        num_identities:  used only if labels='identity'.  If
            num_identities=1000, then all patients get identified as coming
            from precisely 1 of 1000 bins.
        epoch_size:  If defined, randomly sample without replacement N images each epoch.

    Returns:
        (
        {'train_dset': ..., 'val_dset': ..., 'test_dset': ...,
         'train_loader': ..., 'val_loader': ..., 'test_loader': ...
         },

        ('Pneumonia', 'Cardiomegaly', ...)  # class names defined by `labels`
        )
    """
    _label_cleanup_dct = dict(D.CheXpert.LABEL_CLEANUP_DICT)
    if labels == 'identity':
        class_names = list(range(num_identities))
        get_ylabels = lambda dct: (
            D.CheXpert.format_labels(dct, labels=['index'], as_tensor=True) % num_identities
        ).long()
    else:
        if labels == 'diagnostic':
            class_names = D.CheXpert.LABELS_DIAGNOSTIC
        elif labels == 'leaderboard':
            class_names = D.CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD
        else:
            class_names = [x.replace('_', ' ') for x in labels.split(',')]
            assert all(x in D.CheXpert.LABELS_ALL for x in class_names), f'unrecognized class names: {labels}'
        for k in class_names:
            if k in D.CheXpert.LABELS_DIAGNOSTIC:
                _label_cleanup_dct[k][np.nan] = 0  # remap missing value to negative
        get_ylabels = lambda dct: \
            D.CheXpert.format_labels(dct, labels=class_names, as_tensor=True).float()
    kws = dict(
        getitem_transform=lambda dct: (dct['image'], get_ylabels(dct)),
        label_cleanup_dct=_label_cleanup_dct,
        img_loader='cv2',
    )
    if small:
        kls = D.CheXpert_Small
        kws['img_transform'] = tvt.Compose([
            tvt.ToTensor(),
            tvt.CenterCrop((320,320)) if small else (lambda x: x),
        ])
    else:
        kls = D.CheXpert
        kws['img_transform'] = tvt.Compose([
            tvt.ToTensor(),
            #  lambda x: x.to('cuda', non_blocking=True),  # assume num_workers=0
            ResizeCenterCropTo((2320, 2320))  # preserving aspect ratio and center crop
        ])

    train_dset = kls(use_train_set=True, **kws)
    # split the dataset into train and val sets
    # ensure patient images exist only in one set.  no mixing.
    train_idxs, val_idxs = group_random_split(
        [train_frac, val_frac], group=train_dset.labels_csv['Patient'].values)
    val_dset = torch.utils.data.Subset(train_dset, val_idxs)
    train_dset = torch.utils.data.Subset(train_dset, train_idxs)
    # the 200 sample "test" set
    test_dset = kls(use_train_set=False, **kws)
    # data loaders
    batch_size = int(os.environ.get('batch_size', 15))
    print('batch size', batch_size)
    batch_dct = dict(
        #  collate_fn=_upsample_pad_minibatch_imgs_to_same_size,
        #  pin_memory=True,
        num_workers=int(os.environ.get("num_workers", 0)))  # upsample pad must take time
    print('num workers', batch_dct['num_workers'])
    train_loader=DataLoader(
        train_dset, batch_size=batch_size,
        sampler=RandomSampler(train_dset, epoch_size or len(train_dset)), **batch_dct)
    val_loader=DataLoader(val_dset, batch_size=batch_size, **batch_dct)
    test_loader=DataLoader(test_dset, batch_size=batch_size, **batch_dct)
    #
    # debugging:  vis dataset
    #  from deepfixcx.plotting import plot_img_grid
    #  from matplotlib import pyplot as plt
    #  plt.ion()
    #  fig, ax = plt.subplots(1,2)
    #  print('hello world')
    #  for mb in train_loader:
    #      plot_img_grid(mb[0].squeeze(1), num=1, suptitle=f'shape: {mb[0].shape}')
    #      plt.show(block=False)
    #      plt.pause(1)
    #  #
    #  import sys ; sys.exit()
    return (dict(
        train_dset=train_dset, val_dset=val_dset, test_dset=test_dset,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    ), class_names)


def get_dset_intel_mobileodt(stage_trainval:str, use_val:str, stage_test:str, augment:str
                             ) -> tuple[dict[str,Optional[Union[Dataset,DataLoader]]], list[str]]:
    """Obtain train/val/test splits for the IntelMobileODT Cervical Cancer
    Colposcopy dataset, and the data loaders.

    Args:
        stage_trainval: the `stage` for training and validation.
            i.e. Possible choices:  {'train', 'train+additional'}
            Train / val split is 70/30 random stratified split.
        use_val: Whether to create a validation set
            Choices:  {"val", "noval"}
        stage_test: the `stage` for test set.  Should be "test".
        augment: Type of augmentations to apply.  One of {'v1', }.
            "v1" - make the aspect ratio .75, resize images to (200,150), and convert in range [0,1]
    Returns:
        (
        {'train_dset': ..., 'val_dset': ..., 'test_dset': ...,
         'train_loader': ..., 'val_loader': ..., 'test_loader': ...
         },

        ('Type 1', 'Type 2', 'Type 3')
        )
    """
    assert augment == 'v1', 'code bug: other augmentations not implemented'
    base_dir = './data/intel_mobileodt_cervical_resized'
    dset_trainval = D.IntelMobileODTCervical(stage_trainval, base_dir)
    _y = [dset_trainval.getitem(i, load_img=False)
          for i in range(len(dset_trainval))]
    dct = {'test_dset': D.IntelMobileODTCervical(stage_test, base_dir)}
    if use_val == 'noval':
        dct['train_dset'] = dset_trainval
        dct['val_dset'] = None
    else:
        assert use_val == 'val', f'unrecognized option: {use_val}'
        idxs_train, idxs_val = list(
            StratifiedShuffleSplit(1, test_size=.3).split(
                np.arange(len(dset_trainval)), _y))[0]
        dct['train_dset'] = torch.utils.data.Subset(dset_trainval, idxs_train)
        dct['val_dset'] = torch.utils.data.Subset(dset_trainval, idxs_val)

    # preprocess train/val/test images all the same way
    preprocess_v1 = tvt.Compose([
        #
        ### previously done (to save computation time) ###
        #  D.IntelMobileODTCervical.fix_aspect_ratio,
        #  tvt.Resize((200, 150)),  # interpolation=tvt.InterpolationMode.NEAREST),
        #
        lambda x: x.float()/255.
    ])
    dct = {k: D.PreProcess(v, lambda xy: (
        preprocess_v1(xy[0]),
        #  onehot(xy[1].unsqueeze(0).long()-1, 3).squeeze_().float()))
        xy[1].long()-1))
        for k,v in dct.items()}
    dct.update(dict(
        train_loader=DataLoader(dct['train_dset'], batch_size=20, shuffle=True),
        test_loader=DataLoader(dct['test_dset'], batch_size=20),))
    if dct['val_dset'] is None:
        dct['val_loader'] = None
    else:
        dct['val_loader'] = DataLoader(dct['val_dset'], batch_size=20)
    class_names = [x.replace('_', ' ') for x in D.IntelMobileODTCervical.LABEL_NAMES]
    return dct, class_names


def get_dset_flowers102(preprocess_fn=None):
    dct = dict(
        train_dset=tvd.Flowers102(
            'data/flowers102', split='train', download=True),
        val_dset=tvd.Flowers102(
            'data/flowers102', split='val', download=True),
        test_dset=tvd.Flowers102(
            'data/flowers102', split='test', download=True),
    )
    # from MDMLP paper's default settings
    # https://github.com/Amoza-Theodore/MDMLP/blob/main/ymls/flowers102_sgd.yml
    kws = dict(
        input_size=(3, 224, 224),
        batch_size=int(os.environ.get('batch_size', 16)),
        scale=(.08, 1.0),
        ratio=(.75, 1.33),
        hflip=.5,
        color_jitter=.4,
        mean=(0.4330, 0.3819, 0.2964),
        std=(0.2945, 0.2464, 0.2732),
        num_workers=int(os.environ.get('num_workers', 3)),
        crop_pct=.875,
        persistent_workers=(0 != int(os.environ.get('num_workers', 3)))
    )
    dct.update(
        train_loader=timm.data.loader.create_loader(
            dct['train_dset'], is_training=True, **kws),
        val_loader=timm.data.loader.create_loader(
            dct['val_dset'], is_training=False, **kws),
        test_loader=timm.data.loader.create_loader(
            dct['test_dset'], is_training=False, **kws),
    )
    # batch_size=15
    # dct.update(
        # train_loader=DataLoader(
        #     dct['train_dset'], batch_size=batch_size, shuffle=True),
        # val_loader=DataLoader(
        #     dct['val_dset'], batch_size=batch_size, shuffle=False),
        # test_loader=DataLoader(
        #     dct['test_dset'], batch_size=batch_size, shuffle=False),
    # )
    class_names: List[str] = list(str(x) for x in range(102))
    return dct, class_names


def get_dset_food101():
    dct = dict(
        train_dset=tvd.Food101(
            'data/food101', split='train', download=True),
        val_dset=None,  # tvd.Food101(
            # 'data/food101', split='val', download=True),
        test_dset=tvd.Food101(
            'data/food101', split='test', download=True),
    )
    # from MDMLP paper's default settings
    # https://github.com/Amoza-Theodore/MDMLP/blob/main/ymls/food101_sgd.yml
    kws = dict(
        input_size=(3, 224, 224),
        batch_size=int(os.environ.get('batch_size', 16)),
        crop_pct=.875,
        mean=(0.5450, 0.4435, 0.3436),
        std=(0.2729, 0.2758, 0.2798),
        scale=(.08, 1.0),
        ratio=(.75, 1.33),
        hflip=.5,
        color_jitter=.4,
        num_workers=int(os.environ.get('num_workers', 3)),
        persistent_workers=(0 != int(os.environ.get('num_workers', 3))),
    )
    dct.update(
        train_loader=timm.data.loader.create_loader(
            dct['train_dset'], is_training=True, **kws),
        val_loader=None,  # timm.data.loader.create_loader(
            # dct['val_dset'], is_training=False, **kws),
        test_loader=timm.data.loader.create_loader(
            dct['test_dset'], is_training=False, **kws),
    )
    class_names: List[str] = list(str(x) for x in range(102))
    return dct, class_names


class Food101Transform_DeepFixCXImg2Img(torch.nn.Module):
    """Convert ToTensor and apply DeepFixCXImg2Img and run on cuda GPU"""
    def __init__(self, C, J, P):
        super().__init__()
        self.to_tensor = tvt.ToTensor()
        self.transform = tvt.Compose([
            tvt.Resize((512, 512)),
            DeepFixCXImg2Img(
                C, J, P, restore_orig_size=True,).cuda(),
            tvt.Resize((256, 256)),
            tvt.CenterCrop((224, 224))])
        # self.topil = tvt.ToPILImage()

    def forward(self, img):
        img = self.to_tensor(img).cuda()
        img = self.transform(img.unsqueeze_(0)).squeeze_(0)
        # img = self.topil(img)
        # return (img) * 255  # if using timm dataloader
        return (img)


def get_dset_food101_deepfixcxed(J: int, P: int):
    """Hack to do deepfixcx to food101 dataset. remove the timm transforms
    Not trying to improve speed since model is fixed at 224
    """
    dct, class_names = get_dset_food101()
    for td in [dct['train_dset'], dct['test_dset']]:
        td.transform = Food101Transform_DeepFixCXImg2Img(3, J, P)
    kws = dict(
        num_workers=int(os.environ.get('num_workers', 3)),
        persistent_workers=(0 != int(os.environ.get('num_workers', 3))),
        batch_size=int(os.environ.get('batch_size', 16)))
    dct['train_loader'] = DataLoader(dct['train_dset'], shuffle=True, **kws)
    dct['test_loader'] = DataLoader(dct['test_dset'], shuffle=False, **kws)
    return dct, class_names
