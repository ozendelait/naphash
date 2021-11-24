import torch
from torchvision.models import resnet34
from fastai.vision.all import cnn_learner, accuracy, CSVLogger
from fastai.vision.all import ProgressCallback
from fastai.data.transforms import RegexLabeller
from fastai.vision.all import aug_transforms, Normalize, DataBlock, ImageBlock, CategoryBlock, Resize, IndexSplitter
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from torch import Tensor as pyttensor, cuda as pytcuda

#necessary for recorder metrics printout
import fastai_metrics 
from fastai_metrics import silent_progress, plot_confusion_matrix, FocalLoss

import contextlib
import matplotlib.pyplot as plt
import os, glob
from collections import Counter
import numpy as np
import json

def files_in_subdirs(start_dir, pattern = ["*.png","*.jpg","*.jpeg"]):
    files = []
    for p in pattern:
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir,p)))
    return files

cif_cats = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cif_col = {'airplane':'c', 'automobile':'r', 'bird':'m', 'cat':'k', 'deer':'orange', 'dog':'y', 'frog':'limegreen', 'horse':'saddlebrown', 'ship':'b', 'truck':'darkgreen'}
cif_sets = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'B0', 'B1', 'B2', 'b0', 'b1', 'b2', 'i0', 'i1', 'i2', 'I0', 'I1', 'I2', 'm0', 'm1', 'm2', 'm3']

def get_cat(p):
    return p.split('/')[-2]
def count_per_cat(cifar10_paths, idx_train=10000):
    return Counter([get_cat(p) for p in cifar10_paths[idx_train:]])

def reduce_to(cifar10_paths, max_num=5000, per_cat={}, multiple_cat={}, idx_train=10000):
    train_paths_cat, cnt_num = {}, {}
    for p in cifar10_paths[idx_train:]:
        cat = get_cat(p)
        if cnt_num.get(cat,0) >= per_cat.get(cat,max_num):
            continue
        train_paths_cat.setdefault(cat,[]).append(p)
        cnt_num[cat] = cnt_num.get(cat,0)+1
    train_paths = []
    for c in train_paths_cat.keys():
        for _ in range(multiple_cat.get(c,1)):
            train_paths += train_paths_cat[c]
    return cifar10_paths[:idx_train]+train_paths

def reduce_paths(list_id="r0", max_num=5000, per_cat={}, multiple_cat={}, idx_train=10000):
    sets = json.load(open('cifar10_hashsets.json','r'))
    train_paths_cat = {}
    train_paths = []
    for c,l in sets[list_id].items():
        train_paths_cat[c]=[sets['paths'][i] for i in l[:int(per_cat.get(c,max_num))]]
    for c in cif_cats:
        for _ in range(multiple_cat.get(c,1)):
            train_paths += train_paths_cat[c]
    return sets['paths'][:idx_train]+train_paths

def train_cifar10(cifar10_paths, 
                  idx_train=10000, # the first x cifar10_paths should be validation paths; remainder training
                  model=resnet34, 
                  epochs_per_pass = 10, 
                  bs = 512,  #resnet34 w. bs 512 fits easily on Titan RTX (22GB of RAM); larger bs do not increase speed
                  img_dim_load = 128,  # this is larger than the input 32x32; givings augmentations a larger chance to pad/augment
                  img_dim_cif = 96, # this is also larger than the input; giving model larger receptive field)
                  verbose = True, # verbose output including data visualizations
                  weights_per_class = None, #if set, uses focal loss with these weights-per-class
                  device_ids = None # list of GPU device ids
                  ):  
    
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
    if not weights_per_class is None:
        #using FocalLoss with per-class weighting
        vocab_inv = {v:k for k,v in ods_train.vocab.o2i.items()}
        all_weights = [weights_per_class.get(vocab_inv[i],1.0) for i in range(len(vocab_inv))]
        weight_cel= pyttensor([1.] * np.array(all_weights))
        if pytcuda.is_available():
            weight_cel = weight_cel.cuda()
        learn = cnn_learner(ods_train, model, metrics=accuracy, pretrained=True, loss_func=FocalLoss(weight=weight_cel))
    else:   
        learn = cnn_learner(ods_train, model, metrics=accuracy, pretrained=True)
    
    # learn.lossfunc = CrossEntropyFlat(weight = Tensor([1.] * data.c).cuda())
    if device_ids:
        learn.model = torch.nn.DataParallel(learn.model, device_ids=device_ids)
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