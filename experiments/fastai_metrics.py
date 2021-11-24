#based on code from Ignacio Oguiza at https://forums.fast.ai/t/plotting-metrics-after-learning/69937/3
from fastai.imports import patch, delegates, use_kwargs_dict
from fastai.torch_core import subplots
from fastai.learner import *
from  fastai.data.all import CrossEntropyLossFlat

from torch.nn import CrossEntropyLoss
from torch import exp as pytexp

from sklearn.metrics import confusion_matrix

import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, **kwargs):
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]
    add_lr = 'lr' in self.hps
    if add_lr: 
        names += ['lr']
    n = len(names) - 1
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        if add_lr and i == len(names)-1:
            ax.plot(self.hps['lr'], color='#1f77b4', label='train')
            #max_idx, max_epoch = len(self.hps['lr']), len(self.values)
            #ticks = [(f*max_epoch)/max_idx for f in range(max_idx)]
            #ax.set_xticklabels(ticks)  
        else:
            ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    plt.show()

#based on fastai code https://github.com/fastai/fastai/blob/master/fastai/interpret.py
def plot_confusion_matrix(pred, true_y,  vocab={}, normalize=False, title='Confusion matrix', cmap="Blues", norm_dec=2,
                          plot_txt=True, **kwargs):
    "Plot the confusion matrix, with `title` and using `cmap`."
    # This function is mainly copied from the sklearn docs
    cm = confusion_matrix(true_y, pred)
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(vocab))
    plt.xticks(tick_marks, vocab, rotation=90)
    plt.yticks(tick_marks, vocab, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

    ax = fig.gca()
    ax.set_ylim(len(vocab)-.5,-.5)

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)

# adapted from https://github.com/fastai/fastai/blob/52d6302eb4487e86382e663ef5c10ee950c07ad1/fastai/losses.py
class FocalLoss(CrossEntropyLoss):
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop('reduction') if 'reduction' in kwargs else 'mean'
        super().__init__(*args, reduction='none', **kwargs)
    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = pytexp(-ce_loss)
        fl_loss = (1-pt)**self.gamma * ce_loss
        return fl_loss.mean() if self.reduce == 'mean' else fl_loss.sum() if self.reduce == 'sum' else fl_loss
class FocalLossFlat(CrossEntropyLossFlat):
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop('reduction') if 'reduction' in kwargs else 'mean'
        super().__init__(*args, reduction='none', axis=axis, **kwargs)
    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = pytexp(-ce_loss)
        fl_loss = (1-pt)**self.gamma * ce_loss
        return fl_loss.mean() if self.reduce == 'mean' else fl_loss.sum() if self.reduce == 'sum' else fl_loss
    
# still needs to be enclosed in "with contextlib.redirect_stdout(None):" to work
from fastai.vision.all import Learner, Recorder, ProgressCallback
import fastprogress
from functools import partial
class silent_progress():
    ''' Context manager to disable the progress update bar and Recorder print'''
    def __init__(self,learn:Learner):
        print(' ', end='', flush=True) #hack; see https://github.com/tqdm/tqdm/issues/485
        self.learn = learn
        self.prev_recorder = None
        
    def __enter__(self):
        fastprogress.fastprogress.NO_BAR = True
        self.learn.remove_cb(ProgressCallback)
        if hasattr(self.learn, 'recorder') and hasattr(self.learn.recorder, 'silent'):
            self.prev_recorder = self.learn.recorder.silent
            self.learn.recorder.silent = True
        return self.learn
    
    def __exit__(self,type,value,traceback):
        self.learn.recorder.silent = self.prev_recorder
        