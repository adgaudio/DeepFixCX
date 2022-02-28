import glob
import pandas as pd
import numpy as np
import torch as T
import re
import os
from os.path import dirname, abspath, basename
import argparse as ap
import sklearn.metrics

from deepfix.train import match, DSETS
from simplepytorch.datasets import CheXpert


def evaluate_perf(model, loader, device, class_names):
    model.to(device, non_blocking=True)
    model.eval()
    y = []
    yhat = []
    with T.no_grad():
        for minibatch in loader:
            X = minibatch[0].to(device, non_blocking=True)
            _y = minibatch[1].to(device, non_blocking=True)
            _yhat = model(X)
            _yhat = (_yhat.sigmoid() )
            yhat.append(_yhat)
            y.append(_y)
    y = T.cat(y, 0)
    yhat = T.cat(yhat, 0)
    # get the ROC AUC per class
    rv = {}
    for i, cls in enumerate(class_names):
        # ignore uncertain groundtruth
        assert len(y[:,i].unique()) <= 2, 'need to implement: ignore uncertain labels'
        if y[:,i].sum(0) > 0 and (1-y)[:,i].sum(0) > 0:
            rv[f'ROC AUC {cls}'] = sklearn.metrics.roc_auc_score(
                y[:,i].view(-1).cpu().numpy(), yhat[:,i].view(-1).cpu().numpy())
        else:
            rv[f'ROC AUC {cls}'] = np.nan
    # get the ROC AUC for leaderboard if possible
    leaderboard_classes = [
        n for n, kls in enumerate(class_names)
        if kls in CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD]
    if len(leaderboard_classes) == len(CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD):
        rv['ROC AUC LEADERBOARD'] = sklearn.metrics.roc_auc_score(
            y[:, leaderboard_classes].cpu().numpy(),
            yhat[:, leaderboard_classes].cpu().numpy(), average='macro')
    else:
        rv['ROC AUC LEADERBOARD'] = np.nan
    return rv


if __name__ == "__main__":
    par = ap.ArgumentParser()
    par.add_argument(
        'runid_regex', type=re.compile,
        help='regular expression matching the experiment id')
    par.add_argument('--device', default='cuda')
    args = par.parse_args()

    N_epochs = 50

    fps = glob.glob(f'results/*/checkpoints/epoch_{N_epochs}.pth')
    assert len(fps), 'no results at ./results/*/checkpoints/epoch_{N_epochs}.pth'
    fps = [x for x in fps if re.search(args.runid_regex, x)]
    assert len(fps), "didn't find any results"

    results = []
    for fp in fps:
        print(fp)
        csv_fp = f'{dirname(dirname(fp))}/roc_auc.csv'
        if os.path.exists(csv_fp):
            print('loading from file: {csv_fp}')
            df = pd.read_csv(csv_fp)
        else:
            # get the model
            model_dct = T.load(fp, map_location=args.device)
            # get the dataset loader that was used
            log_fp = list(sorted(glob.glob(f'{dirname(dirname(abspath(fp)))}/*console.log')))[-1]
            with open(log_fp, 'r') as fin:
                fin.readline()
                cmdline = fin.readline()
                dset_spec = re.search('--dset (chexpert_small:.*?|chexpert_small15k:.*?) ', cmdline).group(1)
            dset_dct, class_names = match(dset_spec, DSETS)
            # compute performance
            res_dct = evaluate_perf(model_dct['model'], dset_dct['test_loader'], args.device, class_names)
            # get the experiment id
            experiment_id = basename(dirname(dirname(fp)))
            # save the results to experiments directory
            df = pd.DataFrame(
                res_dct, index=pd.Index([(experiment_id, N_epochs)], name=('run_id', 'epoch')))
        df.to_csv(csv_fp)
        # aggregated results
        results.append(df)
    df = pd.concat(results)
    print(df.to_string())

    os.makedirs('./results/plots/roc_auc', exist_ok=True)
    df.to_csv(f'./results/plots/roc_auc/test_{args.runid_regex.pattern}.csv')
    ax = df.plot.barh(legend=False)
    ax.figure.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.figure.savefig(
        f'./results/plots/roc_auc/test_{args.runid_regex.pattern}.png', bbox_inches='tight')
