import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

import sys, os
sys.path.append('.')
sys.path.append('..')

from torch.autograd import Variable
from torch.utils import data
from torch import optim

import subprocess
import yaml

losses_softmax = []
losses_triplet = []
losses_rec = []
losses_mean = []
accuracies_all = []
accuracies_known = []
accuracies_novel = []


losses = [losses_softmax, losses_triplet, losses_rec, losses_mean]
losses_id = ["loss_softmax", "loss_triplet", "loss_rec", "loss_mean"]

accuracies = [accuracies_all, accuracies_known, accuracies_novel]
accuracies_id = ["accuracies_all", "accuracies_known", "accuracies_novel"]

steps = []
accuracy_step = []

ID = ""

args_local = None
ckpt_full_path =  ""


def save_checkpoint(epoch, model, optimizer, description):
    global ckpt_full_path

    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'optimizer_state' : optimizer.state_dict(),}

    ckpt_full_path = "{}/{}_{}_{}_{}.pkl".format( args_local.ckpt_path, args_local.arch, args_local.dataset, description, args_local.id)
    torch.save(state, ckpt_full_path)

    # print(f"Saved model state to: {ckpt_full_path}")

def show_setup(args, n_classes, optimizer, loss_fn):
    global args_local, ID
    args_local=args

    print("Model: {} | Training on: {} | Number of Classes: {}".format(args.arch, args.dataset, n_classes))
    print("Embedding size: {}".format(args.embedding_size))
    print("Epochs: {}".format(args.n_epoch))
    print("Optimizer: {}".format(optimizer))
    print("Loss function: {}".format(loss_fn))

    ID = "Model: {} \n Training on: {} \n Number of Classes: {} \n Epochs: {} \n Optimizer: {} \n Loss function: {} ".format(args.arch, 
        args.dataset, n_classes,args.n_epoch, optimizer, loss_fn)

def get_instances(args):

    #global args_local
    #args_local=args

    gt_file = 'loader/dataset_splits.yml'

    #gt_file = '../loader/dataset_splits.yml' #../ for local

    with open(gt_file, 'r') as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)

    instances = doc[args.dataset][args.instances]

    return instances

def log_loss(epoch, total_epochs, step, loss_softmax=None, loss_triplet=None, loss_rec=None, loss_mean=None):
    global steps, losses_softmax, losses_triplet, losses_rec, losses_mean, losses, args_local, ID

    if loss_softmax != None: losses_softmax.append(loss_softmax)
    if loss_triplet != None: losses_triplet.append(loss_triplet)
    if loss_rec != None: losses_rec.append(loss_rec)
    if loss_mean != None: losses_mean.append(loss_mean)

    steps.append(step)

    log_message = "" 

    for idx in range(len(losses)):

        if len(losses[idx]) > 0:
    
            log_message += losses_id[idx] + ": %.4f | " % losses[idx][-1] 

    print("Epoch [{}/{}] | Global step: {} | {} ".format(epoch+1, total_epochs, step, log_message))

    if step % args_local.logs_freq == 0:
        np.savez("{}/{}_{}_train_log_{}".format(args_local.ckpt_path, args_local.dataset, args_local.arch, args_local.id),  steps=steps, losses_softmax=losses_softmax, 
            losses_triplet=losses_triplet, losses_rec=losses_rec, losses_mean=losses_mean, ID=ID)

def eval_model(unknown_ratio, repeat_num, step, sets, softmax_enabled):
    global ckpt_full_path
    global args_local
    global accuracies_all, accuracies_known, accuracies_novel, accuracies, ID, accuracy_step

    sets = sets

    if sets == "all": sets_to_test = ["full", "known", "novel"]
    elif sets == "known": sets_to_test = ["known"]
    elif sets == "novel": sets_to_test = ["novel"]
    elif sets == "full": sets_to_test = ["full"]

    for set_to_test in sets_to_test:
        run_str =  f"--ckpt_path={ckpt_full_path} "
        run_str += f"--dataset={args_local.dataset} "
        run_str += f"--softmax_enabled={1 if softmax_enabled else 0} "
        run_str += f"--instances={set_to_test} "
        run_str += f"--embedding_size={args_local.embedding_size} "
        run_str += f"--unknown_ratio={unknown_ratio} "
        run_str += f"--repeat_num={repeat_num} "

        valid_run_str = "python3 test/test_embeddings.py --split=valid " + run_str
        train_run_str = "python3 test/test_embeddings.py --split=train " + run_str

        # Get the embeddings of the validation and training set
        subprocess.call([valid_run_str], shell=True)
        subprocess.call([train_run_str], shell=True)
        
        #print (ckpt_full_path)
        if set_to_test == "full":
            accuracy_all = subprocess.check_output(["python3 test/nearest_neighbours.py --ckpt_path={} --instances={}".format(ckpt_full_path, set_to_test)], shell=True)
            accuracy_all = float(accuracy_all.decode('utf-8'))
            accuracies_all.append(accuracy_all)
            print ("Accuracy full: {} | step: {} ".format(accuracy_all, step))

        elif set_to_test == "known":
            accuracy_known = subprocess.check_output(["python3 test/nearest_neighbours.py --ckpt_path={} --instances={}".format(ckpt_full_path, set_to_test)], shell=True)
            accuracy_known = float(accuracy_known.decode('utf-8'))
            accuracies_known.append(accuracy_known)
            print ("Accuracy known: {} | step: {} ".format(accuracy_known, step)) 
             
        elif set_to_test == "novel":
            accuracy_novel = subprocess.check_output(["python3 test/nearest_neighbours.py --ckpt_path={} --instances={}".format(ckpt_full_path, set_to_test)], shell=True)
            accuracy_novel = float(accuracy_novel.decode('utf-8'))
            accuracies_novel.append(accuracy_novel) 
            print ("Accuracy novel: {} | step: {} ".format(accuracy_novel, step))


    accuracy_step.append(step)
    #print ("OK")

    np.savez("{}/{}_{}_accuracies_log_{}".format(args_local.ckpt_path, args_local.dataset, args_local.arch, args_local.id),  steps=accuracy_step, accuracies_all=accuracies_all, 
            accuracies_known=accuracies_known, accuracies_novel=accuracies_novel, ID=ID)

    return accuracy_all
