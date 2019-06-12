import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np

LABELS = ['SGD', 'AdaGrad', 'Adam', 'AMSGrad', 'AdaBound', 'AMSBound']


def get_curve_data(model='ResNet', dataset='cifar10'):
    folder_path = 'curve_more_net'
    filenames = [name for name in os.listdir(folder_path) if name.startswith(model) and '-'+dataset+'-' in name]
    paths = [os.path.join(folder_path, name) for name in filenames]
    keys = [name.split('-')[-1] for name in filenames]
    return {key: torch.load(fp) for key, fp in zip(keys, paths)}


def plot(model='ResNet', dataset='cifar10', optimizers=None, curve_type='train'):
    # assert model in ['ResNet', 'DenseNet'], 'Invalid model name: {}'.format(model)
    assert curve_type in ['train', 'test'], 'Invalid curve type: {}'.format(curve_type)
    assert all(_ in LABELS for _ in optimizers), 'Invalid optimizer'

    curve_data = get_curve_data(model=model, dataset=dataset)

    plt.figure()
    plt.title('{} Accuracy for {} on {}'.format(curve_type.capitalize(), model, dataset))
    plt.xlabel('Epoch')
    plt.ylabel('{} Accuracy %'.format(curve_type.capitalize()))
    if dataset == 'cifar10' and curve_type == 'test':
        ymin, ymax = 70, 101
    elif dataset == 'cifar10' and curve_type == 'train':
        ymin, ymax = 82.5, 101
    elif dataset == 'cifar100' and curve_type == 'test':
        ymin, ymax = 45, 80
    elif dataset == 'cifar100' and curve_type == 'train':
        ymin, ymax = 60, 101
    plt.ylim(ymin, ymax)

    for optim in optimizers:
        linestyle = '--' if 'Bound' in optim else '-'
        accuracies = np.array(curve_data[optim.lower()]['{}_acc'.format(curve_type)])
        plt.plot(accuracies, label=optim, ls=linestyle)

    plt.grid(ls='--')
    plt.legend()
    # plt.show()
    plt.savefig(f'figs/{model}-{dataset}-{curve_type}.png')
    plt.cla()

plot(model='ResNet18', dataset='cifar10', optimizers=LABELS, curve_type='train')
plot(model='ResNet18', dataset='cifar10', optimizers=LABELS, curve_type='test')
plot(model='ResNet50', dataset='cifar10', optimizers=LABELS, curve_type='train')
plot(model='ResNet50', dataset='cifar10', optimizers=LABELS, curve_type='test')
plot(model='ResNet101', dataset='cifar10', optimizers=LABELS, curve_type='train')
plot(model='ResNet101', dataset='cifar10', optimizers=LABELS, curve_type='test')
plot(model='DenseNet169', dataset='cifar10', optimizers=LABELS, curve_type='train')
plot(model='DenseNet169', dataset='cifar10', optimizers=LABELS, curve_type='test')

plot(model='ResNet18', dataset='cifar100', optimizers=LABELS, curve_type='train')
plot(model='ResNet18', dataset='cifar100', optimizers=LABELS, curve_type='test')
plot(model='ResNet50', dataset='cifar100', optimizers=LABELS, curve_type='train')
plot(model='ResNet50', dataset='cifar100', optimizers=LABELS, curve_type='test')
plot(model='ResNet101', dataset='cifar100', optimizers=LABELS, curve_type='train')
plot(model='ResNet101', dataset='cifar100', optimizers=LABELS, curve_type='test')
plot(model='DenseNet169', dataset='cifar100', optimizers=LABELS, curve_type='train')
plot(model='DenseNet169', dataset='cifar100', optimizers=LABELS, curve_type='test')
