import numpy as np
import os
import time
import importlib
from typing import Dict, Callable, List
import tqdm
import sys

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
import json
from script.evaluation import performance_metric

if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    topk_l = [10]
    # first is the fixed name suffix, second is the variable suffix
    method_config_m = {
        'dessert': {
            'build_index': {
                'n_table': 32,
            },
            'retrieval': [
                {'initial_filter_k': 32, "nprobe_query": 4, 'remove_centroid_dupes': True, "n_thread": 8},
            ]
        },
        'plaid': {
            'build_index': {},
            'retrieval': [
                {'ndocs': 32, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
            ]
        }
    }

    dessert_build_index_suffix = f'n_table_{method_config_m["dessert"]["build_index"]["n_table"]}'
    dessert_retrieval_suffix_m = {
        topk: [
            f'initial_filter_k_{retrieval_config["initial_filter_k"]}-nprobe_query_{retrieval_config["nprobe_query"]}-' \
            f'remove_centroid_dupes_{retrieval_config["remove_centroid_dupes"]}-n_thread_{retrieval_config["n_thread"]}'
            for retrieval_config in method_config_m['dessert']['retrieval']
        ] for topk in topk_l}

    performance_metric.count_accuracy_by_baseline(username=username, dataset=dataset, topk_l=topk_l,
                                                  method_name='dessert',
                                                  build_index_suffix=dessert_build_index_suffix,
                                                  retrieval_suffix_m=dessert_retrieval_suffix_m)

    plaid_build_index_suffix = f''
    plaid_retrieval_suffix_m = {
        topk: [
            f'ndocs_{retrieval_config["ndocs"]}-ncells_{retrieval_config["ncells"]}-' \
            f'centroid_score_threshold_{"{:.2f}".format(retrieval_config["centroid_score_threshold"])}-' \
            f'n_thread_{retrieval_config["n_thread"]}'
            for retrieval_config in method_config_m['plaid']['retrieval']
        ] for topk in topk_l}

    performance_metric.count_accuracy_by_baseline(username=username, dataset=dataset, topk_l=topk_l,
                                                  method_name='plaid',
                                                  build_index_suffix=plaid_build_index_suffix,
                                                  retrieval_suffix_m=plaid_retrieval_suffix_m)
