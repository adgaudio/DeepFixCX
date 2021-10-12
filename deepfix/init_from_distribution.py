from simple_parsing import ArgumentParser
import torch as T


def init_from_hist_(model:T.nn.Module, hist:dict[str, (T.Tensor, T.Tensor)]):
    """
    Initialize model weights with values from the histogram.
    Modify model inplace.

    Args:
        model: a pytorch model with named parameters
        hist: a dict of histograms of form {"named_parameter": (counts, bin_edges)}
            where "named_parameter" identifies a parameter
            (i.e. from `model.named_parameters()`),
            and where `counts` and `bin_edges` define one histogram for each scalar
            value in the parameter.
            - `counts` is of shape (N, M) where N is the number of scalar
                values in the parameter, and M is the bin size.
            - `bin_edges` is of shape (N+1) denoting the beginning and end of
              each histogram bin.
    """
    for param_name in hist:
        counts, bin_edges = hist[param_name]
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        param = model.get_parameter(param_name)
        # convert histogram to cumulative probability distribution
        cs = counts.cumsum(1)
        cs /= cs[:,[-1]]
        assert T.allclose(cs[:,-1], T.tensor(1.))
        # for each weight parameter, sample a value from the distribution
        sample = T.rand((counts.shape[0], 1))
        csbins = T.searchsorted(cs, sample)
        values = bin_centers[csbins]
        # add noise, staying within boundary of a bin
        bin_width = bin_edges[1] - bin_edges[0]
        noise = T.rand(counts.shape[0], 1, device=bin_width.device) * bin_width/2
        values += noise
        param.data[:] = values.reshape(param.shape)


def reset_optimizer(opt_spec:str, model:T.nn.Module) -> T.optim.Optimizer:
    spec = opt_spec.split(':')
    kls = getattr(T.optim, spec[0])
    params = [(x,float(y)) for x,y in [kv.split('=') for kv in spec[1:]]]
    optimizer = kls(model.parameters(), **dict(params))
    return optimizer


if __name__ == "__main__":
    import sys
    from deepfix.train import TrainOptions, train_config


    def get_cfg():
        p = ArgumentParser()
        p.add_arguments(TrainOptions, dest='TrainOptions')
        args = p.parse_args([
            '--device', 'cuda', '--model', 'resnet18:untrained:3:3']).TrainOptions
        cfg = train_config(args)
        return cfg, args
    # load model
    cfg, args = get_cfg()

    fp = sys.argv[1]  # results/1.I3/histograms/hist_nth_most_salient_resnet18:untrained:3:3.pth
    fp_out = sys.argv[2]  # results/1.I3/hist_nth_most_salient_resnet18:untrained:3:3.csv
    hist = T.load(fp)

    perfs = []
    #  perfs.append(cfg.evaluate_perf(cfg))
    #  print('initialize model')
    #  init_from_hist_(cfg.model, hist)
    #  cfg.optimizer = reset_optimizer(args.opt, cfg.model)
    print('evaluate model')
    perfs.append(cfg.evaluate_perf(cfg, loader=cfg.train_loader))
    for _ in range(5):
        cfg.train_one_epoch(cfg)
        perfs.append(cfg.evaluate_perf(cfg))
        print(perfs[-1])

    import pandas as pd
    df = pd.DataFrame(perfs)
    print(df.to_string())
    df.to_csv(fp_out, index=False)





    #  T.load('./hist_nth_weight_resnet18:untrained:3:3.pth
    # eval, train+eval