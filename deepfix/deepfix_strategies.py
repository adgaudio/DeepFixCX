


class DeepFix_TrainOneEpoch:
    """
    DeepFix: Re-initialize bottom `P` fraction of least salient weights
    every `N` epochs.  Wraps a `train_one_epoch` function.

    Example Usage:
        >>> using_deepfix = DeepFixTrainingStrategy(
                N=3, P=.2, train_one_epoch_fn=TL.train_one_epoch)
        >>> cfg = TrainConfig(..., train_one_epoch=using_deepfix)
        >>> cfg.train()
    """
    def __init__(self, N:int, P:float, R:int,
                 train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result]):
        """
        Args:
            N: Re-initialize weights every N epochs.  N>1.
            P: The fraction of least salient weights to re-initialize.  0<=P<=1.
            R: Num repetitions.  How many times to re-initialize in a row.
            train_one_epoch_fn: The function used to train the model.
                Expects as input a TL.TrainConfig and outputs a TL.Result.
        """
        assert 0 <= P <= 1, 'sanity check: 0 <= P <= 1'
        assert N >= 1, 'sanity check: N > 1'
        assert R >=1
        self.N, self.P, self.R = N, P, R
        self._wrapped_train_one_epoch = train_one_epoch_fn
        self._counter = 0

    def __call__(self, cfg:TL.TrainConfig):
        self._counter = (self._counter + 1) % self.N
        if self._counter % self.N == 0:
            print('DeepFix REINIT')
            for i in range(self.R):
                reinitialize_least_salient(
                    costfn_multiclass,
                    #  lambda y, yh: yh[y].sum(),  # TODO: wrong
                    model=cfg.model, loader=cfg.train_loader,
                    device=cfg.device, M=50, frac=self.P*((self.R-i)/self.R),
                    opt=cfg.optimizer
                )
        return self._wrapped_train_one_epoch(cfg)


@dc.dataclass
class DeepFix_LambdaInit:
    init_fn: Callable[T.nn.Module,None]
    args:tuple = ()
    train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result] = TL.train_one_epoch
    _called = False

    def __call__(self, cfg: TL.TrainConfig):
        if not self._called:
            self._called = True
            print(f"DeepFix {self.init_fn.__name__}{self.args}")
            self.init_fn(cfg.model, *self.args)
        return self.train_one_epoch_fn(cfg)


@dc.dataclass
class DeepFix_DHist:
    fp: str
    train_one_epoch_fn:Callable[TL.TrainConfig, TL.Result] = TL.train_one_epoch
    init_with_hist:bool = True
    fixed:bool = False
    _called = False

    def __call__(self, cfg: TL.TrainConfig):
        if not self._called:
            self._called = True
            if self.init_with_hist:
                hist = T.load(self.fp)
                print('DeepFix HISTOGRAM Initialization')
                init_from_hist_(cfg.model, hist)
            if self.fixed:
                for layer in cfg.model.modules():
                    layer.requires_grad_(False)
                for name,x in list(cfg.model.named_modules())[::-1]:
                    if len(list(x.parameters())):
                        x.requires_grad_(True)
                        break
                print(f"DeepFix:  all layers fixed ...except layer: {name}")
        return self.train_one_epoch_fn(cfg)
