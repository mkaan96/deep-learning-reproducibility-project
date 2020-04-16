import os
import argparse
import pathlib
import numpy as np
import torch
import json

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from LazyDataset import LazyDataset
from utilities import log, valid_seed, load_batch_gcnn
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process(model, dataloader, top_k, criterion, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        batch = load_batch_gcnn(batch, device)
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
        batched_states = (c, ei, ev, v, torch.sum(n_cs), torch.sum(n_vs))  # prevent padding
        batch_size = n_cs.shape

        if optimizer:
            model.zero_grad()
            model.train()
            logits = model(batched_states) # training mode
            logits = torch.unsqueeze(torch.squeeze(logits, 0)[cands.long()], 0) # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now

            loss = criterion(logits, best_cands.long())
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.squeeze(logits, 0)[cands.type(torch.LongTensor)], 0) # filter candidate variables
            logits = model.pad_output(logits, n_cands.type(torch.LongTensor))  # apply padding now
            
            loss = criterion(logits, best_cands.long())

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
        default='data/samples/setcover/500r_1000c_0.05d'
    )

    parser.add_argument(
        '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'setcover-small', 'mik'],
        default='setcover'
    )

    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=valid_seed,
        default=0,
    )

    parser.add_argument(
        '--lr',
        help='Initial learning rate',
        type=float,
        default=0.001,
    )

    parser.add_argument(
        '--optimizer',
        help='Optimizer to use',
        default='Adam',
        choices=['Adam', 'RMSprop']
    )

    args = parser.parse_args()
    with open('config.json', 'r') as f:
        config = json.load(f)
    ### HYPER PARAMETERS ###
    max_epochs = 2
    epoch_size = 312
    batch_size = 32
    valid_batch_size = config['valid_batch_size']
    lr = args.lr
    patience = 10
    early_stopping = 20
    top_k = [1, 3, 5, 10]
    train_ncands_limit = np.inf
    valid_ncands_limit = np.inf

    if args.lr == 0.001:
        lr_dir = 'lr-normal'
    elif args.lr > 0.001:
        lr_dir = 'lr-high'
    else:
        lr_dir = 'lr-low'

    running_dir = f"trained_models/{args.problem}/baseline/{args.seed}/{lr_dir}/{args.optimizer}"

    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"seed {args.seed}", logfile)
    log(f"optimizer: {args.optimizer}")
    log(f"problem: {args.problem}")
    log(f"samples_path: {args.samples_path}")

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
    model = NeuralNet(device).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise Exception('Invalid optimizer')

    # Should set lr *= 0.2 when .step() is called
    lr_scheduler = ExponentialLR(optimizer, 0.2)
    ### TRAINING LOOP ###
    best_loss = np.inf
    plateau_count = 0
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        if epoch > 0:
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = LazyDataset(epoch_train_files)
            train_data = DataLoader(train_data, batch_size=batch_size)
            train_loss, train_kacc = process(model, train_data, top_k, criterion, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        with torch.no_grad():
            valid_loss, valid_kacc = process(model, valid_data, top_k, criterion, None)
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
                lr_scheduler.step()
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr_scheduler.get_lr()}", logfile)

    model = NeuralNet(device).to(device)
    model.load_state_dict(torch.load(os.path.join(running_dir, 'best_params.pkl')))
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    with torch.no_grad():
        valid_loss, valid_kacc = process(model, valid_data, top_k, criterion, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
