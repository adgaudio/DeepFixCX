import glob
import pandas as pd
import numpy as np
import torch as T
import re
import os
from os.path import dirname, abspath, basename
import argparse as ap
import sklearn.metrics

from deepfixcx.train import match, DSETS
from simplepytorch.datasets import CheXpert


def get_class_thresholds(model, loader, device, class_names):
    with T.no_grad():
        y, yhat = [], []
        for minibatch in loader:
            X = minibatch[0].to(device, non_blocking=True)
            _y = minibatch[1].to(device, non_blocking=True)
            _yhat = model(X)
            _yhat = (_yhat.sigmoid() )
            yhat.append(_yhat)
            y.append(_y)
    y = T.cat(y, 0)
    yhat = T.cat(yhat, 0)

    class_thresholds = {}
    for i, cls in enumerate(class_names):
        _y = y[:,i].view(-1).cpu().numpy()
        _yh = yhat[:,i].view(-1).cpu().numpy()
        mask = (_y == 0) | (_y == 1)  # ignore uncertain labels
        fpr, tpr, _thresholds = sklearn.metrics.roc_curve(_y[mask], _yh[mask])
        # assume we want a threshold that gives a maximum probability of correct
        # prediction, or 1 == TP/P == TN / N.  As a maxmization, we get argmax(tp/t
        # + tn/n), which is equal to argmax(tpr - fpr).
        class_thresholds[cls] = _thresholds[np.argmax(tpr - fpr)]
    return class_thresholds


def evaluate_perf(model, loader, device, class_thresholds):
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
    for i, (cls, thresh) in enumerate(class_thresholds.items()):
        # ignore uncertain groundtruth
        assert len(y[:,i].unique()) <= 2, 'need to implement: ignore uncertain labels'
        if y[:,i].sum(0) > 0 and (1-y)[:,i].sum(0) > 0:
            rv[f'ROC AUC {cls}'] = sklearn.metrics.roc_auc_score(
                y[:,i].view(-1).cpu().numpy(), yhat[:,i].view(-1).cpu().numpy())
        else:
            rv[f'ROC AUC {cls}'] = np.nan
        rv[f'BAcc {cls}'] = sklearn.metrics.balanced_accuracy_score(
            y[:, i].view(-1).cpu().numpy(),
            (yhat[:, i] > thresh).view(-1).cpu().numpy())
    # get the ROC AUC for leaderboard if possible
    # (some models don't predict the leaderboard classes)
    leaderboard_class_idxs = [
        n for n, kls in enumerate(class_thresholds)
        if kls in CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD]
    leaderboard_class_names = [
        kls for n, kls in enumerate(class_thresholds)
        if kls in CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD]
    if len(leaderboard_class_idxs) == len(CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD):
        rv['ROC AUC LEADERBOARD'] = sklearn.metrics.roc_auc_score(
            y[:, leaderboard_class_idxs].cpu().numpy(),
            yhat[:, leaderboard_class_idxs].cpu().numpy(), average='macro')
        rv['BAcc LEADERBOARD'] = np.mean([rv[f'BAcc {cls}'] for cls in leaderboard_class_names])
    else:
        rv['ROC AUC LEADERBOARD'] = np.nan
        rv['BAcc LEADERBOARD'] = np.nan
    return rv


if __name__ == "__main__":
    par = ap.ArgumentParser()
    par.add_argument(
        'runid_regex', type=re.compile,
        help='regular expression matching the experiment id')
    par.add_argument('--device', default='cuda')
    par.add_argument('--checkpoint_fname', default='epoch_80.pth')
    par.add_argument('--overwrite', action='store_true',
                     help="Don't re-use existing results.  Re-compute them.")
    args = par.parse_args()

    fps = glob.glob(f'results/*/checkpoints/{args.checkpoint_fname}')
    assert len(fps), 'no results at ./results/*/checkpoints/{args.checkpoint_fname}.pth'
    fps = [x for x in fps if re.search(args.runid_regex, x)]
    assert len(fps), "didn't find any results"

    results = []
    for fp in fps:
        print(fp)
        csv_fp = f'{dirname(dirname(fp))}/roc_auc_{args.checkpoint_fname}.csv'
        if os.path.exists(csv_fp) and not args.overwrite:
            print(f'loading from file: {csv_fp}')
            df = pd.read_csv(csv_fp)
        else:
            # get the model
            model_dct = T.load(fp, map_location=args.device)
            # ... fix backwards incompatibility with old models
            try:
                model_dct['model'].compression_mdl.wavelet_encoder.adaptive = 0
            except:
                pass
            # get the dataset loader that was used
            log_fp = list(sorted(glob.glob(f'{dirname(dirname(abspath(fp)))}/*console.log')))[-1]
            with open(log_fp, 'r') as fin:
                fin.readline()
                cmdline = fin.readline()
                dset_spec = re.search('--dset (chexpert_small:.*?|chexpert_small15k:.*?) ', cmdline).group(1)
            dset_dct, class_names = match(dset_spec, DSETS)
            # get the thresholds for Balanced Accuracy
            class_thresholds = get_class_thresholds(model_dct['model'], dset_dct['val_loader'], args.device, class_names)
            # compute performance
            res_dct = evaluate_perf(model_dct['model'], dset_dct['test_loader'], args.device, class_thresholds)
            # get the experiment id
            experiment_id = basename(dirname(dirname(fp)))
            # save the results to experiments directory
            df = pd.DataFrame(
                res_dct, index=pd.Index([(experiment_id, args.checkpoint_fname)], name=('run_id', 'checkpoint')))
            df.to_csv(csv_fp)
            pd.Series(class_thresholds).to_csv(f'{dirname(dirname(fp))}/class_thresholds.csv')
        # aggregated results
        results.append(df)
    df = pd.concat(results)
    print(df.to_string())

    os.makedirs('./results/plots/roc_auc', exist_ok=True)
    df.to_csv(f'./results/plots/roc_auc/test_{args.runid_regex.pattern}_{args.checkpoint_fname}.csv')

    #  ax = df.plot.barh(legend=False)
    #  ax.figure.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #  ax.figure.savefig(
        #  f'./results/plots/roc_auc/test_{args.runid_regex.pattern}.png', bbox_inches='tight')
