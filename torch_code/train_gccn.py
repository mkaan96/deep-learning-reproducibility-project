import os
import argparse
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import torch
from torch import nn
from torch.utils.data import DataLoader

import utilities
from models.baseline.model import GCNPolicy
from torch_code.LazyDataset import LazyDataset
from torch_code.utilities import log

from torch_code.utilities_tf import load_batch_gcnn
from torch_code.model import NeuralNet
import json


def pretrain(model, dataloader):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """

    model.pre_train_init()

    i = 0
    while True:
        for batch in dataloader:
            batch = load_batch_gcnn(batch)
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
            batched_states = (c, ei, ev, v, n_cs, n_vs)

            if not new_model.pre_train(batched_states):
                break

        res = new_model.pre_train_next()
        if res is None:
            break

        i += 1

    return i


def process(model, dataloader, top_k, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        batch = load_batch_gcnn(batch)
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
        batched_states = (c, ei, ev, v, torch.sum(n_cs, keepdim=True), torch.sum(n_vs, keepdim=True))  # prevent padding
        batch_size = len(n_cs.numpy())

        if optimizer:
            with tf.GradientTape() as tape:
                logits = model(batched_states) # training mode
                logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
                loss = tf.losses.sparse_softmax_cross_entropy(labels=best_cands, logits=logits)
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
        else:
            logits = model(batched_states)  # eval mode
            logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
            loss = tf.losses.sparse_softmax_cross_entropy(labels=best_cands, logits=logits)

        true_scores = model.pad_output(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)
        true_scores = true_scores.numpy()
        true_bestscore = true_bestscore.numpy()

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(logits, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_loss += loss.numpy() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'setcover-small'],
        default='setcover-small'
    )

    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()
    with open('../config.json', 'r') as f:
        config = json.load(f)
    ### HYPER PARAMETERS ###
    max_epochs = 1000
    epoch_size = 312
    batch_size = config['batch_size']
    pretrain_batch_size = config['pretrain_batch_size']
    valid_batch_size = config['valid_batch_size']
    lr = 0.001
    patience = 10
    early_stopping = 20
    top_k = [1, 3, 5, 10]
    train_ncands_limit = np.inf
    valid_ncands_limit = np.inf

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'setcover-small': 'setcover-small/500r_1000c_0.05d'
    }
    problem_folder = problem_folders[args.problem]

    running_dir = f"trained_models/{args.problem}/baseline/{args.seed}"

    os.makedirs(running_dir, exist_ok=True)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    ### NUMPY / TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    train_files = list(pathlib.Path(f'../data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'../data/samples/{problem_folder}/valid').glob('sample_*.pkl'))


    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = LazyDataset(valid_files)
    valid_data = DataLoader(valid_data, batch_size=valid_batch_size)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]

    pretrain_data = LazyDataset(pretrain_files)
    pretrain_data = DataLoader(pretrain_data, batch_size=pretrain_batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model = NeuralNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)
    model = GCNPolicy()
    ### TRAINING LOOP ###
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # TRAIN
        if epoch == 0:
            n = pretrain(model=new_model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            # bugfix: tensorflow's shuffle() seems broken...
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = LazyDataset(epoch_train_files)
            train_data = DataLoader(train_data, batch_size=batch_size)
            train_loss, train_kacc = process(new_model, train_data, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(new_model, valid_data, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_data, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
