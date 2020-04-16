import os
import argparse
import csv
import numpy as np
import time
import pathlib
import torch
import json

from torch.utils.data import DataLoader
from LazyDataset import LazyDataset
from model import NeuralNet
from utilities import load_batch_gcnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch

        pred_scores = policy['model']((c, ei, ev, v, torch.sum(n_cs), torch.sum(n_vs)))

        # filter candidate variables
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
        default='data/samples/setcover/500r_1000c_0.05d'
    )

    parser.add_argument(
        '--problem',
        help='MILP instance type to process.',
        choices=['setcover', 'setcover-small', 'mik'],
        default='setcover'
    )

    parser.add_argument(
        '--lr',
        help='Chosen learning rate',
        choices=['lr-normal', 'lr-high', 'lr-low'],
        default='lr-normal'
    )

    parser.add_argument(
        '--optimizer',
        help='Chosen optimizer',
        choices=['Adam', 'RMSprop'],
        default='Adam'
    )

    args = parser.parse_args()

    print(f"problem: {args.problem}")

    if args.lr == 'lr-normal' and args.optimizer == 'Adam' and args.problem != 'mik':
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [0]
    gcnn_models = ['baseline']

    with open('config.json', 'r') as f:
        config = json.load(f)
    test_batch_size = config['valid_batch_size']

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
            policy['model'].load_state_dict(torch.load(f"trained_models/{args.problem}/baseline/{seed}/{args.lr}/{args.optimizer}/best_params.pkl"))

            test_data = LazyDataset(test_files)
            test_data = DataLoader(test_data, batch_size=test_batch_size)

            policy['model'].eval()
            with torch.no_grad():
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
