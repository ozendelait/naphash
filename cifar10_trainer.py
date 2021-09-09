import torch
from torchvision.models import resnet34
from fastai2.vision.all import cnn_learner, accuracy, CSVLogger
from fastai2.vision.all import ProgressCallback
from fastai2.data.transforms import RegexLabeller
from fastai2.vision.all import aug_transforms, Normalize, DataBlock, ImageBlock, CategoryBlock, Resize, IndexSplitter
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
#necessary for recorder metrics printout
import fastai_metrics 
from fastai_metrics import silent_progress, plot_confusion_matrix

import contextlib
import matplotlib.pyplot as plt
import os, glob
from collections import Counter
import numpy as np

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

def get_cat(p):
    return p.split('/')[-2]

def reduce_to(cifar10_paths, max_num=5000, per_cat={}, idx_train=10000):
    train_paths, cnt_num = [], {}
    for p in cifar10_paths[idx_train:]:
        cat = get_cat(p)
        if cnt_num.get(cat,0) >= per_cat.get(cat,max_num):
            continue
        train_paths.append(p)
        cnt_num[cat] = cnt_num.get(cat,0)+1
    return cifar10_paths[:idx_train]+train_paths

def train_cifar10(cifar10_paths, 
                  idx_train=10000, # the first x cifar10_paths should be validation paths; remainder training
                  model=resnet34, 
                  epochs_per_pass = 10, 
                  bs = 512,  #resnet34 w. bs 512 fits easily on Titan RTX (22GB of RAM); larger bs do not increase speed
                  img_dim_load = 128,  # this is larger than the input 32x32; givings augmentations a larger chance to pad/augment
                  img_dim_cif = 96, # this is also larger than the input; giving model larger receptive field)
                  verbose = True):  
    
    get_cls = RegexLabeller(pat = r'.*/(.*)/')
    #default fastai augmentations (reflective padding, horizontal flipping, up to 10 deg rotation;)
    #see https://docs.fast.ai/vision.augment.html#aug_transforms
    #parameters based on https://gist.github.com/rlandingin/e09e2e568e964466fc3b5634bf18d87a
    tfms = [*aug_transforms(size=img_dim_cif, min_scale=0.8, max_zoom=1.1, max_lighting=0.4),
            Normalize.from_stats(mean=[0.4914 , 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159])]

    #creating a DataBlock 'from_folder' would work but we want to exclude or dublicate images later
    # (i.e. redundancy) -> this can be easily achived using file paths
    dblock = DataBlock(
      blocks=(ImageBlock, CategoryBlock),      # one image input and one categorical target
      splitter=IndexSplitter(range(0,idx_train)),
      batch_tfms=tfms,
      item_tfms=[Resize(img_dim_cif)],
      get_items = lambda x:(cifar10_paths),
      get_y     =  get_cls)
    ods_train = dblock.dataloaders('./',bs=bs)
    
    bad_valid = [t for t in ods_train.valid_ds.items if not '/test/' in t] #just a precaution to make sure validation frames are not in the training set
    assert len(bad_valid) == 0 or len(ods_train.valid_ds.items) != idx_train, "There are invalid validation frames in the dataloader"
    if verbose:
        print("Using: ", len(ods_train.train_ds.items), len(ods_train.valid_ds.items), len(bad_valid))
        ods_train.show_batch(max_n=9 , figsize=(5,5))
        plt.show()
    learn = cnn_learner(ods_train, model, metrics=accuracy, pretrained=True)
    learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1])
    for epoch_passes in [epochs_per_pass,epochs_per_pass]:
        if not verbose:
            learn.recorder.silent = True
            with contextlib.redirect_stdout(None):
                with silent_progress(learn) as learn:
                    learn.fit_one_cycle(epochs_per_pass) # per default all but the last two layers are frozen
        else:
            learn.fit_one_cycle(epochs_per_pass) # per default all but the last two layers are frozen
        if verbose:
            learn.recorder.plot_metrics()
        learn.unfreeze() # unfreezes all layers
    
    learn.remove_cb(ProgressCallback) #necessary due to strange bug (see https://github.com/fastai/fastprogress/issues/72)
    res = learn.tta(n=10)
    #calculate per-class accuracy
    vocab_inv = {v:k for k,v in ods_train.vocab.o2i.items()}
    val_trg = [t.item() for t in res[1]]
    val_res = [t.argmax().item() for t in res[0]]
    
    corr_pred = {'all':0}
    for i in range(idx_train):
        if val_trg[i] == val_res[i]:
            corr_pred[val_trg[i]] = corr_pred.get(val_trg[i],0)+1
            corr_pred['all'] += 1
    count_idx = Counter(val_trg)
    acc_per_class = {vocab_inv[c]:corr_pred.get(c,0)/count_idx[c] for c in count_idx}
    # the test set has an equal distribution of all classes -> mean over classes == mean over all
    # acc_per_class['all'] = corr_pred['all']/idx_train
    
    if verbose:
        plot_confusion_matrix(val_res,val_trg,ods_train.vocab.o2i)
        plt.show()
        print("Min/Mean acc. (tta) : %f/%f"%(np.min(list(acc_per_class.values())), np.mean(list(acc_per_class.values()))), acc_per_class)
    return acc_per_class