import numpy as np
import os
import sys
import json
import pandas as pd
import copy
from typing import Callable

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric, evaluation_pipeline


def count_accuracy(username: str, dataset: str, topk: int,
                   method_id_m: dict,
                   mrr_gnd: tuple, success_gnd: tuple, recall_gnd_id_m: dict):
    has_mrr_groundtruth, mrr_groundtruth_m, mrr_passageID_l = mrr_gnd
    has_success_groundtruth, success_gnd_m, success_passageID_local2global_l, success_passage_l = success_gnd

    gnd_id_m = recall_gnd_id_m

    recall_l = performance_metric.count_recall(gnd_id_m=gnd_id_m, est_id_m=method_id_m, topk=topk)
    recall_m = {
        'recall_p5': '{:.3f}'.format(np.percentile(recall_l, 5)),
        'recall_p50': '{:.3f}'.format(np.percentile(recall_l, 50)),
        'recall_p95': '{:.3f}'.format(np.percentile(recall_l, 95)),
        'recall_mean': '{:.3f}'.format(np.average(recall_l)),
    }

    mrr_l, success_l = None, None
    mrr_m, success_m = {}, {}
    if has_mrr_groundtruth:
        mrr_l = performance_metric.count_mrr(est_id_m=method_id_m, end2end_gnd_m=mrr_groundtruth_m,
                                             end2end_passageID_l=mrr_passageID_l)
        e2e_recall_l = performance_metric.count_end2end_recall(est_id_m=method_id_m, end2end_gnd_m=mrr_groundtruth_m,
                                                               end2end_passageID_l=mrr_passageID_l)
        ndcg_l = performance_metric.count_ndcg(est_id_m=method_id_m, end2end_gnd_m=mrr_groundtruth_m,
                                               end2end_passageID_l=mrr_passageID_l, topk=topk)
        mrr_m = {
            'mrr_p5': '{:.3f}'.format(np.percentile(mrr_l, 5)),
            'mrr_p50': '{:.3f}'.format(np.percentile(mrr_l, 50)),
            'mrr_p95': '{:.3f}'.format(np.percentile(mrr_l, 95)),
            'mrr_max': '{:.3f}'.format(np.percentile(mrr_l, 100)),
            'mrr_mean': '{:.3f}'.format(np.average(mrr_l)),
            'e2e_recall_p5': '{:.3f}'.format(np.percentile(e2e_recall_l, 5)),
            'e2e_recall_p50': '{:.3f}'.format(np.percentile(e2e_recall_l, 50)),
            'e2e_recall_p95': '{:.3f}'.format(np.percentile(e2e_recall_l, 95)),
            'e2e_recall_max': '{:.3f}'.format(np.percentile(e2e_recall_l, 100)),
            'e2e_recall_mean': '{:.3f}'.format(np.average(e2e_recall_l)),
            'ndcg_p5': '{:.3f}'.format(np.percentile(ndcg_l, 5)),
            'ndcg_p50': '{:.3f}'.format(np.percentile(ndcg_l, 50)),
            'ndcg_p95': '{:.3f}'.format(np.percentile(ndcg_l, 95)),
            'ndcg_max': '{:.3f}'.format(np.percentile(ndcg_l, 100)),
            'ndcg_mean': '{:.3f}'.format(np.average(ndcg_l)),
        }
    if has_success_groundtruth:
        success_l = performance_metric.count_success(est_id_m=method_id_m, gnd_queryID2answer_m=success_gnd_m,
                                                     passageID_local2global_l=success_passageID_local2global_l,
                                                     passage_l=success_passage_l)
        success_m = {
            'success_p5': '{:.3f}'.format(np.percentile(success_l, 5)),
            'success_p50': '{:.3f}'.format(np.percentile(success_l, 50)),
            'success_p95': '{:.3f}'.format(np.percentile(success_l, 95)),
            'success_max': '{:.3f}'.format(np.percentile(success_l, 100)),
            'success_mean': '{:.3f}'.format(np.average(success_l)),
        }
    if not has_mrr_groundtruth and not has_success_groundtruth:
        mrr_m = {}
        success_m = {}
    search_accuracy_m = {**recall_m, **mrr_m, **success_m}

    return recall_l, mrr_l, success_l, search_accuracy_m


def approximate_solution_retrieval_outter(username: str, dataset: str,
                                          topk: int, ):
    query_l, queryID_l = evaluation_pipeline.load_query(username=username, dataset=dataset)

    mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                topk=topk)

    embedding_dir = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Embedding/{dataset}'
    method_id_m = performance_metric.read_method_tsv(base_dir=embedding_dir, dataset=dataset, method_name='groundtruth',
                                                     topk=topk, build_index_suffix='',
                                                     retrieval_suffix='')

    recall_l, mrr_l, success_l, search_accuracy_m = count_accuracy(
        username=username, dataset=dataset, topk=topk,
        method_id_m=method_id_m,
        mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)

    build_index_suffix = ''
    retrieval_suffix = ''
    retrieval_info_m = {'n_query': len(queryID_l), 'topk': topk,
                        'search_accuracy': search_accuracy_m}
    method_performance_name = f'{dataset}-retrieval-BruteForce-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/performance'
    performance_filename = os.path.join(result_performance_path, method_performance_name)
    with open(performance_filename, "w") as f:
        json.dump(retrieval_info_m, f)

    print("#############final result###############")
    print("filename", method_performance_name)
    print("search accuracy", retrieval_info_m['search_accuracy'])
    print("########################################")

    final_result_l = []
    final_result_l.append({'filename': method_performance_name,
                           'search_accuracy': retrieval_info_m['search_accuracy']})

    for final_result in final_result_l:
        print("#############final result###############")
        print("filename", final_result['filename'])
        print("search accuracy", final_result['search_accuracy'])
        print("########################################")


if __name__ == '__main__':
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            'dataset_l': ['lotte', 'hotpotqa', 'msmacro', 'quora'],
            'topk_l': [10, 100],
        },
        'local': {
            'username': 'bianzheng',
            # 'dataset_l': ['fake-normal', 'lotte-500-gnd'],
            'dataset_l': ['lotte-500-gnd'],
            'topk_l': [10],
        }
    }
    host_name = 'local'
    config = config_l[host_name]

    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']

    for dataset in dataset_l:
        for topk in topk_l:
            approximate_solution_retrieval_outter(username=username, dataset=dataset, topk=topk)
