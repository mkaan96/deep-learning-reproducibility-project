import os
import argparse
import csv
import numpy as np
import time
import pathlib

import tensorflow as tf
import tensorflow.contrib.eager as tfe


import utilities
from models.baseline.model import GCNPolicy

from utilities_tf import load_batch_gcnn


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


def padding(output, n_vars_per_sample, fill=-1e8):
    n_vars_max = tf.reduce_max(n_vars_per_sample)

    output = tf.split(
        value=output,
        num_or_size_splits=n_vars_per_sample,
        axis=1,
    )
    output = tf.concat([
        tf.pad(
            x,
            paddings=[[0, 0], [0, n_vars_max - tf.shape(x)[1]]],
            mode='CONSTANT',
            constant_values=fill)
        for x in output
    ], axis=0)

    return output


def process(policy, dataloader, top_k):
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:

        if policy['type'] == 'gcnn':
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch

            pred_scores = policy['model']((c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True)), tf.convert_to_tensor(False))

            # filter candidate variables
            pred_scores = tf.expand_dims(tf.gather(tf.squeeze(pred_scores, 0), cands), 0)

        # padding
        pred_scores = padding(pred_scores, n_cands)
        true_scores = padding(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)

        assert all(true_bestscore.numpy() == np.take_along_axis(true_scores.numpy(), best_cands.numpy().reshape((-1, 1)), axis=1))

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(pred_scores, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores.numpy(), pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore.numpy(), axis=1)))
        kacc = np.asarray(kacc)

        batch_size = int(n_cands.shape[0])
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    os.makedirs("results", exist_ok=True)
    result_file = f"results/{args.problem}_validation_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    #seeds = [0, 1, 2, 3, 4]
    seeds = [0]
    gcnn_models = ['baseline']
    test_batch_size = 16
    top_k = [1, 3, 5, 10]

    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'setcover-small': 'setcover-small/500r_1000c_0.05d'
    }
    problem_folder = problem_folders[args.problem]

    result_file = f"results/{args.problem}_test_{time.strftime('%Y%m%d-%H%M%S')}"

    result_file = result_file + '.csv'
    os.makedirs('results', exist_ok=True)

    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    test_files = list(pathlib.Path(f"data/samples/{problem_folder}/test").glob('sample_*.pkl'))
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
            tf.set_random_seed(rng.randint(np.iinfo(int).max))

            policy = {}
            policy['name'] = policy_name
            policy['type'] = policy_type

            policy['model'] = GCNPolicy()
            policy['model'].restore_state(f"trained_models/{args.problem}/baseline/{seed}/best_params.pkl")
            policy['model'].call = tfe.defun(policy['model'].call, input_signature=policy['model'].input_signature)
            policy['batch_datatypes'] = [tf.float32, tf.int32, tf.float32,
                    tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]
            policy['batch_fun'] = load_batch_gcnn

            test_data = tf.data.Dataset.from_tensor_slices(test_files)
            test_data = test_data.batch(test_batch_size)
            test_data = test_data.map(lambda x: tf.py_func(
                policy['batch_fun'], [x], policy['batch_datatypes']))
            test_data = test_data.prefetch(2)

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
