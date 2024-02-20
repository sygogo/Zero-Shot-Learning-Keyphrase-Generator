import json
import os
import pickle
import numpy as np
from numpy.lib._iotools import str2bool

import utility
from utility import stem_extraction, phrase_stmmer, stem_extraction_
import pandas as pd
import argparse


def calculate_micro(result_dict: dict, task_name: str, top_k: str):
    result_present_total_num = 0
    result_absent_total_num = 0
    label_present_num = 0
    label_absent_num = 0
    all_present_num = 0
    all_absent_num = 0
    num = 0
    for index, key in enumerate(result_dict):
        if key == "prompts":
            continue
        values = result_dict[key]
        result_phrases = values[f"{task_name}"]
        label_present = values['label_present']
        label_absent = values['label_absent']

        if top_k == '5':
            num = 5
        if top_k == 'm':
            num = len(label_absent) + len(label_present)
        if top_k == 'all':
            num = len(result_phrases)

        label_present_stem = [stem_extraction(item, phrase_stmmer).strip().replace('-', ' ').replace('_', ' ') for item
                              in
                              label_present]
        label_absent_stem = [stem_extraction(item, phrase_stmmer).strip().replace('-', ' ').replace('_', ' ') for item
                             in
                             label_absent]
        result_phrases_stem = [stem_extraction(item, phrase_stmmer).strip().replace('-', ' ').replace('_', ' ') for item
                               in
                               result_phrases]
        result_phrases_stem_num = result_phrases_stem[:num]

        result_phrases_stem_num_present = set(result_phrases_stem_num) & set(label_present_stem)
        result_phrases_stem_num_absent = set(result_phrases_stem_num) & set(label_absent_stem)
        result_present_total_num += len(result_phrases_stem_num_present)
        result_absent_total_num += len(result_phrases_stem_num_absent)

        all_present_num += min(num, len(result_phrases))
        all_absent_num += num

        label_present_num += len(label_present)
        label_absent_num += len(label_absent)

    present_precision = result_present_total_num / all_present_num
    present_recall = result_present_total_num / label_present_num
    if present_precision + present_recall != 0:
        present_f1 = (2 * present_precision * present_recall) / (present_precision + present_recall)
    else:
        present_f1 = 0
    absent_precision = result_absent_total_num / all_absent_num
    absent_recall = result_absent_total_num / label_absent_num
    if absent_precision + absent_recall != 0:
        absent_f1 = (2 * absent_precision * absent_recall) / (absent_precision + absent_recall)
    else:
        absent_f1 = 0
    return {f"present_precision_{top_k}": present_precision, f"present_recall_{top_k}": present_recall,
            f"present_f1_{top_k}": present_f1, f"absent_precision_{top_k}": absent_precision,
            f"absent_recall_{top_k}": absent_recall, f"absent_f1_{top_k}": absent_f1}


def calculate_macro(result_dict: dict, task_name: str, top_k: str):
    precision_present_all = 0
    recall_present_all = 0
    precision_absent_all = 0
    recall_absent_all = 0
    num = 0
    count = 0
    for index, key in enumerate(result_dict):
        if str(key).__contains__('scores'):
            continue
        if key == "prompts":
            continue
        if key == "desc":
            continue
        values = result_dict[key]
        result_phrases = values[f"{task_name}"]
        label_present = values['label_present']
        label_absent = values['label_absent']

        if top_k == '5':
            num = 5
        if top_k == 'm':
            num = len(label_absent) + len(label_present)
        if top_k == 'all':
            num = len(result_phrases)

        label_present_stem = [stem_extraction_(item, phrase_stmmer).strip().replace('-', ' ').replace('_', ' ') for item
                              in label_present]
        label_absent_stem = [stem_extraction_(item, phrase_stmmer).strip().replace('-', ' ').replace('_', ' ') for item
                             in label_absent]
        result_phrases_stem = [stem_extraction_(item, phrase_stmmer).strip().replace('-', ' ').replace('_', ' ') for
                               item in result_phrases]
        result_phrases_stem_num = result_phrases_stem[:num]

        tp_present = len(set(result_phrases_stem_num) & set(label_present_stem))
        tp_absent = len(set(result_phrases_stem_num) & set(label_absent_stem))
        tp_absent = tp_absent
        if result_phrases_stem_num:
            # if top_k == '5':
            #     precision_present = tp_present / 5
            #     precision_absent = tp_absent / 5
            # else:
            #     precision_present = tp_present / len(result_phrases_stem_num)
            #     precision_absent = tp_absent / len(result_phrases_stem_num)
            precision_present = tp_present / len(result_phrases_stem_num)
            precision_absent = tp_absent / len(result_phrases_stem_num)
        else:
            precision_present = 0
            precision_absent = 0
        if label_present_stem:
            recall_present = tp_present / len(label_present_stem)
        else:
            recall_present = 0
        if label_absent_stem:
            recall_absent = tp_absent / len(label_absent_stem)
        else:
            recall_absent = 0

        precision_present_all += precision_present
        recall_present_all += recall_present
        precision_absent_all += precision_absent
        recall_absent_all += recall_absent

        count += 1

    present_p_avg = precision_present_all / count
    present_r_avg = recall_present_all / count
    if present_p_avg + present_r_avg != 0:
        present_f1 = (2 * present_p_avg * present_r_avg) / (present_p_avg + present_r_avg)
    else:
        present_f1 = 0

    absent_p_avg = precision_absent_all / count
    absent_r_avg = recall_absent_all / count
    if absent_p_avg + absent_r_avg != 0:
        absent_f1 = (2 * absent_p_avg * absent_r_avg) / (absent_p_avg + absent_r_avg)
    else:
        absent_f1 = 0

    return {f"present_precision_{top_k}": round(present_p_avg, 3), f"present_recall_{top_k}": round(present_r_avg, 3),
            f"present_f1_{top_k}": round(present_f1, 3), f"absent_precision_{top_k}": round(absent_p_avg, 3),
            f"absent_recall_{top_k}": round(absent_r_avg, 3), f"absent_f1_{top_k}": round(absent_f1, 3)}


def calculate_f1_(result_dict, task_name):
    # dict_5 = calculate_micro(result_dict, task_name, '5')
    # dict_m = calculate_micro(result_dict, task_name, 'm')
    # dict_all = calculate_micro(result_dict, task_name, 'all')
    dict_5 = calculate_macro(result_dict, task_name, '5')
    dict_m = calculate_macro(result_dict, task_name, 'm')
    dict_all = calculate_macro(result_dict, task_name, 'all')
    dict_p_5 = dict(list(dict_5.items())[:3])
    dict_a_5 = dict(list(dict_5.items())[3:])
    dict_p_m = dict(list(dict_m.items())[:3])
    dict_a_m = dict(list(dict_m.items())[3:])
    dict_p_all = dict(list(dict_all.items())[:3])
    dict_a_all = dict(list(dict_all.items())[3:])

    return dict_p_5, dict_p_m, dict_p_all, dict_a_5, dict_a_m, dict_a_all


def get_result(dataset_name: str, coefficient: float, only_positive=False):
    files = [item for item in os.listdir(os.path.join("Results", dataset_name)) if item.endswith('pkl') or item.endswith('pkl_')]
    paths = [os.path.join("Results", dataset_name, item) for item in files]
    for path in paths:
        with open(path, 'rb') as pickle_file:
            data_dicts = pickle.load(pickle_file)

        extract_result = calculate_f1_(data_dicts, 'extract')
        # extend_result = calculate_f1_(data_dicts, 'extend')

        print(path)
        if args.verbose:
            # print('retrieve only: ', retrieve_result)
            print('extract only: ', extract_result)
            # print('extend only: ', extend_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficient', type=float, default=1)
    parser.add_argument('--dataset', type=str, default='inspec')
    parser.add_argument('--verbose', type=str2bool, default='true')
    args = parser.parse_args()
    get_result(args.dataset, coefficient=args.coefficient)
