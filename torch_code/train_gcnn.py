import sys
sys.path.insert(0,"../")
import os
import argparse
import pathlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import utilities
from torch_code.LazyDataset import LazyDataset
from torch_code.utilities import log

from torch_code.utilities_tf import load_batch_gcnn, remove_batch_from_memory
from torch_code.model import NeuralNet
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            batch = load_batch_gcnn(batch, device)
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
            batched_states = (c, ei, ev, v, n_cs, n_vs)

            if not model.pre_train(batched_states):
                remove_batch_from_memory(batch)
                torch.cuda.empty_cache()
                break

            remove_batch_from_memory(batch)
            torch.cuda.empty_cache()
        res = model.pre_train_next()
        if res is None:
            break

        i += 1

    return i


def process(model, dataloader, top_k, cross, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        batch = load_batch_gcnn(batch, device)
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
        batched_states = (c, ei, ev, v, torch.sum(n_cs), torch.sum(n_vs))  # prevent padding
        batch_size = n_cs.shape

        if optimizer:
            optimizer.zero_grad()
            model.train()
            logits = model(batched_states) # training mode
            logits = torch.unsqueeze(torch.squeeze(logits, 0)[cands.type(torch.LongTensor)], 0) # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now

            loss = cross(logits, best_cands.long())
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.squeeze(logits, 0)[cands.type(torch.LongTensor)], 0) # filter candidate variables
            logits = model.pad_output(logits, n_cands.type(torch.LongTensor))  # apply padding now
            
            cross = nn.CrossEntropyLoss()
            loss = cross(logits, best_cands.long())

        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, -1, True)
        true_bestscore = true_bestscore[0]
 
        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k)[1]
            pred_top_k_true_scores = true_scores.gather(1, pred_top_k)
            kacc.append(torch.mean(torch.any(torch.eq(pred_top_k_true_scores, true_bestscore), dim=1).float(), dim=0).item())
        kacc = np.asarray(kacc)

        mean_loss += loss.item() * batch_size[0]
        mean_kacc += kacc * batch_size[0]
        n_samples_processed += batch_size[0]

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--samples_path',
        default='../data/samples/setcover-small/500r_1000c_0.05d'
    )

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

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    train_files = list(pathlib.Path(f'{args.samples_path}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'{args.samples_path}/valid').glob('sample_*.pkl'))


    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = LazyDataset(valid_files)
    valid_data = DataLoader(valid_data, batch_size=valid_batch_size)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]

    pretrain_data = LazyDataset(pretrain_files)
    pretrain_data = DataLoader(pretrain_data, batch_size=pretrain_batch_size)

    model = NeuralNet(device).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cross = nn.CrossEntropyLoss().to(device)
    ### TRAINING LOOP ###
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            # bugfix: tensorflow's shuffle() seems broken...
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = LazyDataset(epoch_train_files)
            train_data = DataLoader(train_data, batch_size=batch_size)
            train_loss, train_kacc = process(model, train_data, top_k, cross, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_data, top_k, cross, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

    model = NeuralNet(device).to(device)
    model.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))
    cross = nn.CrossEntropyLoss().to(device)
    valid_loss, valid_kacc = process(model, valid_data, top_k, cross, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
