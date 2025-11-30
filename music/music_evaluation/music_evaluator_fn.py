# Modified from: https://github.com/RichardYang40148/mgeval/blob/master/__main__.py

from argparse import ArgumentParser
import glob
import copy
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pretty_midi
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import random

from music_evaluation.mgeval import core, utils


def delete_nan(arr):
    arr[np.isnan(arr) | np.isinf(arr)] = 0
    return arr


def compute_oa(set1, set2, outdir):
    # set1: source files, set2: gen files
    if not any(set2):
        print("Error: baseline set it empty")
        exit()

    print('Evaluation begins ~')
    if not any(set1):
      print("Error: sample set it empty")
      exit()

    num_samples = min(len(set2), len(set1))

    print("Number of samples in use: ", num_samples)
    evalset = {
                'total_used_pitch': np.zeros((num_samples, 1))
              , 'pitch_range': np.zeros((num_samples, 1))
              , 'avg_IOI': np.zeros((num_samples, 1))
              , 'total_pitch_class_histogram': np.zeros((num_samples, 12))
              , 'mean_note_velocity':np.zeros((num_samples, 1))
              , 'mean_note_duration':np.zeros((num_samples, 1))
              , 'note_density':np.zeros((num_samples, 1))
              }


    # print(evalset)

    metrics_list = list(evalset.keys())

    single_arg_metrics = (
        [ 'total_used_pitch'
        , 'avg_IOI'
        , 'total_pitch_class_histogram'
        , 'pitch_range'
        , 'mean_note_velocity'
        , 'mean_note_duration'
        , 'note_density'
        , 'pitch_class_transition_matrix'
        ])

    set1_eval = copy.deepcopy(evalset)
    set2_eval = copy.deepcopy(evalset)

    sets = [ (set1, set1_eval), (set2, set2_eval) ]


    # Extract Features
    for _set, _set_eval in sets:
        for i in range(0, num_samples):
            feature = core.extract_feature(_set[i])
            for metric in metrics_list:
                evaluator = getattr(core.metrics(), metric)
                if metric in single_arg_metrics:
                    tmp = evaluator(feature)
                else:
                    tmp = evaluator(feature, 0)
                _set_eval[metric][i] = tmp

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))


    # Calculate Intra-set Metrics
    for i, metric in enumerate(metrics_list):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            set1_intra[test_index[0]][i] = utils.c_dist(
                set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
            set2_intra[test_index[0]][i] = utils.c_dist(
                set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

    # Calculate Inter-set Metrics
    for i, metric in enumerate(metrics_list):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metric][test_index], set2_eval[metric])


    plot_set1_intra = np.transpose(
        set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(
        set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(
        sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)

    plot_set1_intra = delete_nan(plot_set1_intra)
    plot_set2_intra = delete_nan(plot_set2_intra)
    plot_sets_inter = delete_nan(plot_sets_inter)

    # output = {}
    df_output = defaultdict(list)
    for i, metric in enumerate(metrics_list):
        print("-----------------------------")
        print('calculating KL and OA of: {}'.format(metric))

        mean = np.mean(set1_eval[metric], axis=0).tolist()
        std = np.std(set1_eval[metric], axis=0).tolist()
        filename = metric+".png"

        kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
        ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
        # kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
        # ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])

        print("KL(set1 || inter_set): ", kl1)
        print("OA(set1, inter1): ", ol1)
        # print("KL(set2 || inter_set): ", kl2)
        # print("OA(set2, inter1): ", ol2)
        # output[metric] = [mean, std, kl1, ol1, kl2, ol2]
        df_output['attribute'].append(metric)
        df_output['KL'].append(kl1)
        df_output['OA'].append(ol1)
    df = pd.DataFrame(df_output)
    avg = {'attribute': 'avg', 'KL': df['KL'].mean(), 'OA': df['OA'].mean()}
    avg_df = pd.DataFrame([avg])
    df = pd.concat([df, avg_df], ignore_index=True)

    print('Evaluation Complete.')
    return df
