import argparse
import datetime
import pickle
import gzip
import numpy as np
import torch


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


def load_batch_gcnn(sample_files, device):
    """
    Loads and concatenates a bunch of samples into one mini-batch.
    """
    c_features = []
    e_indices = []
    e_features = []
    v_features = []
    candss = []
    cand_choices = []
    cand_scoress = []

    # load samples
    for filename in sample_files:
        with gzip.open(filename, 'rb') as f:
            sample = pickle.load(f)

        sample_state, _, sample_action, sample_cands, cand_scores = sample['data']

        sample_cands = np.array(sample_cands)
        cand_choice = np.where(sample_cands == sample_action)[0][0]  # action index relative to candidates

        c, e, v = sample_state
        c_features.append(c['values'])
        e_indices.append(e['indices'])
        e_features.append(e['values'])
        v_features.append(v['values'])
        candss.append(sample_cands)
        cand_choices.append(cand_choice)
        cand_scoress.append(cand_scores)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_cands_per_sample = [cds.shape[0] for cds in candss]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)
    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_vs_per_sample[:-1]
        ], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
        for j, e_ind in enumerate(e_indices)], axis=1)
    # candidate indices as well
    candss = np.concatenate([cands + shift
        for cands, shift in zip(candss, cv_shift[1])])
    cand_choices = np.array(cand_choices)
    cand_scoress = np.concatenate(cand_scoress, axis=0)

    # convert to tensors
    c_features = torch.tensor(c_features, dtype=torch.float32).to(device)
    e_indices = torch.tensor(e_indices, dtype=torch.long).to(device).detach()
    e_features = torch.tensor(e_features, dtype=torch.float32).to(device)
    v_features = torch.tensor(v_features, dtype=torch.float32).to(device)
    n_cs_per_sample = torch.tensor(n_cs_per_sample, dtype=torch.int32).to(device).detach()
    n_vs_per_sample = torch.tensor(n_vs_per_sample, dtype=torch.int32).to(device).detach()
    candss = torch.tensor(candss, dtype=torch.int32).to(device)
    cand_choices = torch.tensor(cand_choices, dtype=torch.int32).to(device)
    cand_scoress = torch.tensor(cand_scoress, dtype=torch.float32).to(device)
    n_cands_per_sample = torch.tensor(n_cands_per_sample, dtype=torch.int32).to(device)

    return c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, n_cands_per_sample, candss, cand_choices, cand_scoress


def print_used_memory():
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    print(f'{ (c/1e9):.3f} GB cached memory | {(a/1e9):.3f} GB allocated memory')
