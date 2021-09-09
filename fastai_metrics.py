#based on code from Ignacio Oguiza at https://forums.fast.ai/t/plotting-metrics-after-learning/69937/3
from fastai2.imports import patch, delegates
from fastai2.torch_core import subplots
from fastai2.learner import *
import math
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

# still needs to be enclosed in "with contextlib.redirect_stdout(None):" to work
from fastai2.vision.all import Learner, Recorder, ProgressCallback
import fastprogress
from functools import partial
class silent_progress():
    ''' Context manager to disable the progress update bar and Recorder print'''
    def __init__(self,learn:Learner):
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