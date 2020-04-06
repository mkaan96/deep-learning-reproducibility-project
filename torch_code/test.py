import os
import argparse
import csv
import numpy as np
import time
import pathlib
import torch
import utilities

from torch.utils.data import DataLoader
from torch_code.LazyDataset import LazyDataset
from torch_code.model import NeuralNet
from torch_code.utilities_tf import load_batch_gcnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_batch_flat(sample_files, feats_type, augment_feats, normalize_feats):
    cand_features = []
    cand_choices = []
    cand_scoress = []

    for i, filename in enumerate(sample_files):
        cand_states, cand_scores, cand_choice = utilities.load_flat_samples(filename, feats_type, 'scores', augment_feats, normalize_feats)

        cand_features.append(cand_states)
        cand_choices.append(cand_choice)
        cand_scoress.append(cand_scores)

    n_cands_per_sample = [v.shape[0] for v in cand_features]

    cand_features = np.concatenate(cand_features, axis=0).astype(np.float32, copy=False)
    cand_choices = np.asarray(cand_choices).astype(np.int32, copy=False)
    cand_scoress = np.concatenate(cand_scoress, axis=0).astype(np.float32, copy=False)
    n_cands_per_sample = np.asarray(n_cands_per_sample).astype(np.int32, copy=False)

    return cand_features, n_cands_per_sample, cand_choices, cand_scoress


def padding(output, n_vars_per_sample, pad_value=-1e8):
    n_vars_max = torch.max(n_vars_per_sample)

    output = torch.split(output, tuple(n_vars_per_sample), 1)

    output2 = []
    for x in output:
        newx = torch.nn.functional.pad(x,(0, n_vars_max.item() - x.shape[1]),'constant', pad_value)
        output2.append(newx)

    output = torch.cat(output2, 0)

    return output


def process(policy, dataloader, top_k):
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        batch = load_batch_gcnn(batch, device)
        pred_scores = None
        cand_scores = None
        n_cands = None
        if policy['type'] == 'gcnn':
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch

            pred_scores = policy['model']((c, ei, ev, v, torch.sum(n_cs), torch.sum(n_vs)))

            # filter candidate variables
            # pred_scores = tf.expand_dims(tf.gather(tf.squeeze(pred_scores, 0), cands), 0)
            pred_scores = torch.unsqueeze(torch.squeeze(pred_scores, 0)[cands.type(torch.LongTensor)], 0)

        # padding
        pred_scores = padding(pred_scores, n_cands)
        true_scores = padding(cand_scores.reshape((1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, -1, True)
        true_bestscore = true_bestscore[0]

        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(pred_scores, k)[1]
            pred_top_k_true_scores = true_scores.gather(1, pred_top_k)
            kacc.append(torch.mean(torch.any(torch.eq(pred_top_k_true_scores, true_bestscore), dim=1).float(), dim=0).item())
        kacc = np.asarray(kacc)

        batch_size = int(n_cands.shape[0])
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc


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
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    args = parser.parse_args()

    print(f"problem: {args.problem}")
    print(f"gpu: {args.gpu}")

    seeds = [0, 1, 2, 3, 4]
    # seeds = [0]
    gcnn_models = ['baseline']
    test_batch_size = 16
    top_k = [1, 3, 5, 10]

    result_file = f"results/{args.problem}_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    os.makedirs('results', exist_ok=True)

    test_files = list(pathlib.Path(f"{args.samples_path}/test").glob('sample_*.pkl'))
    test_files = [str(x) for x in test_files]

    print(f"{len(test_files)} test samples")

    evaluated_policies = [['gcnn', model] for model in gcnn_models]

    fieldnames = [
        'policy',
        'seed',
    ] + [
        f'acc@{k}' for k in top_k
    ]
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        policy_type = 'gcnn'
        policy_name = 'baseline'

        print(f"{policy_type}:{policy_name}...")
        for seed in seeds:
            rng = np.random.RandomState(seed)
            torch.manual_seed(rng.randint(np.iinfo(int).max))

            policy = {}
            policy['name'] = policy_name
            policy['type'] = policy_type

            policy['model'] = NeuralNet(device).to(device)
            policy['model'].load_state_dict(torch.load(f"trained_models/{args.problem}/baseline/{seed}/best_params.pkl"))

            test_data = LazyDataset(test_files)
            test_data = DataLoader(test_data, batch_size=test_batch_size)

            policy['model'].eval()
            test_kacc = process(policy, test_data, top_k)
            print(f"  {seed} " + " ".join([f"acc@{k}: {100*acc:4.1f}" for k, acc in zip(top_k, test_kacc)]))

            writer.writerow({
                **{
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': seed,
                },
                **{
                    f'acc@{k}': test_kacc[i] for i, k in enumerate(top_k)
                },
            })
            csvfile.flush()
