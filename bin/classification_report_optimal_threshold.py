"""
First, find the 'optimal' thresholds for classification with regards to balanced
accuracy.
Then, compute the prediction performance.

This fixes problems that may occur if ROC AUC might be high but classification
performance low.
"""
import sklearn.metrics
import numpy as np
import pandas as pd
import torch as T
import argparse as ap

from deepfixcx.dsets import get_dset_flowers102, get_dset_food101, get_dset_food101_deepfixcxed


def evaluate(model, loader, device):
    model.eval()
    with T.no_grad():
        y, yhat = [], []
        for minibatch in loader:
            X = minibatch[0].to(device, non_blocking=True)
            _y = minibatch[1].to(device, non_blocking=True)
            _yhat = model(X)
            # _yhat = (_yhat.sigmoid() )
            yhat.append(_yhat)
            y.append(_y)
    y = T.cat(y, 0)
    yhat = T.cat(yhat, 0)
    return y, yhat


def get_class_thresholds(model, loader, device, class_names):
    y, yhat = evaluate(model, loader, device)
    #y = y.new_ones(yhat.shape[1])[y].cpu().numpy()

    class_thresholds = {}
    for i, cls in enumerate(class_names):
        _y = y[:, i].view(-1).cpu().numpy()
        _yh = yhat.argmax(1).cpu().numpy()  # yhat[:, i].view(-1).cpu().numpy()
        fpr, tpr, _thresholds = sklearn.metrics.roc_curve(_y, _yh)
        # assume we want a threshold that gives a maximum probability of correct
        # prediction, or 1 == TP/P == TN / N.  As a maxmization, we get argmax(tp/t
        # + tn/n), which is equal to argmax(tpr - fpr).
        class_thresholds[cls] = _thresholds[np.argmax(tpr - fpr)]
    return class_thresholds


def main(args: ap.Namespace):
    model = T.load(args.model_checkpoint, map_location=args.device)['model']

    if args.dset == 'flowers102':
        dct, class_names = get_dset_flowers102()
    elif args.dset.startswith('food101:'):
        J, P = re.match('food101:(\d+):(\d+)', args.dset)
        dct, class_names = get_dset_food101_deepfixcxed(J=int(J), P=int(P))
    elif args.dset == 'food101':
        dct, class_names = get_dset_food101()
    else:
        raise NotImplementedError()

    #_class_thresholds = get_class_thresholds(
    #    model, dct['val_loader'] or dct['train_loader'], args.device, class_names)
    #thresholds = T.tensor(
    #    [_class_thresholds[name] for name in class_names]).unsqueeze(0)

    y, yhat = evaluate(model, dct['test_loader'], args.device)
    print('acc', sklearn.metrics.accuracy_score(y.cpu().numpy(), yhat.argmax(1).cpu().numpy()))
    yhat_class = yhat > thresholds

    df = classification_report(y, yhat, yhat_class)
    print(df)
    if args.savefp:
        df['model_checkpoint'] = args.model_checkpoint
        df['dset'] = args.dset
        df.to_csv(args.savefp, index=False)


def classification_report(y, yhat, yhat_class):
    results = {
        'Test Acc': sklearn.metrics.accuracy_score(y, yhat_class),
        'Test BAcc': sklearn.metrics.balanced_accuracy_score(y, yhat_class),
        'Test Precision': sklearn.metrics.precision_score(y, yhat_class),
        'Test Recall': sklearn.metrics.recall_score(y, yhat_class),
        'Test MacroF1': sklearn.metrics.f1_score(
            y, yhat_class, average='macro'),
        'Test MicroF1': sklearn.metrics.f1_score(
            y, yhat_class, average='micro'),
        'Test MacroROCAUC': sklearn.metrics.roc_auc_score(
            y, yhat, average='macro'),
        'Test MicroROCAUC': sklearn.metrics.roc_auc_score(
            y, yhat, average='micro')
    }
    return pd.DataFrame(results)


def arg_parser():
    par = ap.ArgumentParser()
    aa = par.add_argument
    aa('model_checkpoint', help='filepath to checkpoint')
    aa('--dset', choices=['flowers102'])
    aa('--device', default='gpu')
    aa('--savefp', default='./classification_report_{dset}.csv',
       help='optional path to CSV file')
    return par


if __name__ == '__main__':
    main(arg_parser().parse_args())
