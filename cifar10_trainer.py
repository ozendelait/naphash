import torch
from torchvision.models import resnet34
from fastai2.vision.all import cnn_learner, accuracy, CSVLogger
from fastai2.vision.all import ProgressCallback
from fastai2.data.transforms import RegexLabeller
from fastai2.vision.all import aug_transforms, Normalize, DataBlock, ImageBlock, CategoryBlock, Resize, IndexSplitter
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
#necessary for recorder metrics printout
import fastai_metrics 
from fastai_metrics import silent_progress

import contextlib
import matplotlib.pyplot as plt
import os, glob

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

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
    tta_acc = float(accuracy(*res))
    if verbose:
        print("Accuracy (tta) :", tta_acc)
    return tta_acc