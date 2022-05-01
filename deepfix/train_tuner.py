from functools import partial
from ray import tune
import dataclasses as dc
import simple_parsing as sp
from simplepytorch import raytunelib as RTL
from simplepytorch import trainlib as TL
from typing import Dict, Optional
import torch as T

from deepfix import train


def train_config(hyperparams:Dict, args:'TrainOptions') -> TL.TrainConfig:
    hp = hyperparams
    preprocess = None if not hp['medpool'] else T.nn.Sequential(
            train.MedianPool2d(kernel_size=hp['medpool']['kernel_size'], stride=hp['medpool']['stride'], same=True),
            T.nn.UpsamplingNearest2d((320,320)))
    model = train.QTLineClassifier(
        train.RLine(
            (320,320),
            nlines=int(hp['rlines']),
            hlines=list(range(int(hp['hlines_start']), int(hp['hlines_end']), int(hp['hlines_step']))),
            heart_roi=hp['heart'],
            sum_aggregate=hp['sum'],
            zero_top_frac=0, seed=1,
        ), preprocess)
    model.to(args.device, non_blocking=True)

    opt_spec = args.opt.replace('lr=tune', f'lr={hp["lr"]}')

    optimizer = train.reset_optimizer(opt_spec, model)
    loss_fn = train.match(args.lossfn, train.LOSS_FNS)
    if args.loss_reg != 'none':
        loss_fn = train.RegularizedLoss(model, loss_fn, args.loss_reg)

    return RTL.RTLTrainConfig(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        **train.get_dset_loaders_resultfactory(args.dset, args.device),
        device=args.device,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        experiment_id=args.experiment_id,
        checkpoint_if=TL.CheckpointIf(metric='val_ROC_AUC AVG', mode='max')
    )


@dc.dataclass
class TrainOptions2(train.TrainOptions):
    """Command-line Configuration"""
    raytune: RTL.RTLDefaults = RTL.RTLDefaults(
        num_samples=40, metric='val_ROC_AUC AVG', mode='max',
        resources_per_trial=RTL.ResourcesPerTrial(gpu=.5, cpu=24/2//2))
    mode:str = sp.choice(['debugtrain', 'tune'], default='tune')
    start_epoch:int = 1
    model = None

def main():
    p = sp.ArgumentParser()
    p.add_arguments(TrainOptions2, dest='TrainOptions2')
    args = p.parse_args().TrainOptions2
    print(args)

    hyperparameter_search_space = {
        'lr': tune.uniform(.0005, .0015),
        'rlines': tune.quniform(100, 300, 1),
        'hlines_start': tune.quniform(80, 120, 1),
        'hlines_end': tune.quniform(280, 320, 1),
        'hlines_step': tune.quniform(5, 15, 1),
        'heart': tune.choice([True, False]),
        'sum': tune.choice([True, False]),
        'medpool': tune.choice([None, ])#, {
            #  'kernel_size': tune.choice([3, 6, 12]),
            #  'stride': tune.sample_from(tune.choice([1,3])),
        #  }]),
    }

    if args.mode == 'debugtrain':
        hp = {k: v.sample() for k,v in hyperparameter_search_space.items()}
        print('Hyperparameters:', hp)
        cfg = train_config(hp, args)
        cfg.train(cfg)
    else:
        assert args.mode == 'tune'

        analysis = RTL.ray_train(
            args.experiment_id,
            rtl_args=args.raytune,
            space=hyperparameter_search_space,
            get_train_config=partial(train_config, args=args),
            #
            stop=tune.stopper.TrialPlateauStopper(
                metric=args.raytune.metric, mode=args.raytune.mode,
                grace_period=10, num_results=10, std=.0075),
            verbose=2,
        )
        print("Winning hyperparameters:\n",
              analysis.get_best_config(args.raytune.metric, args.raytune.mode))


if __name__ == "__main__":
    main()
