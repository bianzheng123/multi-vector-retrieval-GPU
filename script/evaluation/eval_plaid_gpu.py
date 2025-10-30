import logging
import numpy as np
import os
import sys
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric
from baseline.ColBERT import run as colbert_run


def retrieval(username: str, dataset: str, retrieval_parameter_l: list, topk: int,
              n_centroid: int, n_bit: int, n_pid_kmeans: int, search_gpu: bool):
    logging.info(f"plaid retrieval start {dataset}")
    colbert_run.retrieval_official(username=username, dataset=dataset,
                                   topk=topk, search_config_l=retrieval_parameter_l,
                                   n_centroid=n_centroid, n_bit=n_bit, n_pid_kmeans=n_pid_kmeans,
                                   search_gpu=search_gpu)
    logging.info(f"plaid retrieval end {dataset}")


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for ndocs in grid_search_para['ndocs']:
        for ncells in grid_search_para['ncells']:
            for centroid_score_threshold in grid_search_para['centroid_score_threshold']:
                for n_thread in grid_search_para['n_thread']:
                    parameter_l.append(
                        {"ndocs": ndocs, "ncells": ncells,
                         "centroid_score_threshold": centroid_score_threshold,
                         "n_thread": n_thread})
    return parameter_l


if __name__ == '__main__':
    # 'ndocs': searcher.config.ndocs,
    # 'ncells': searcher.config.ncells,
    # 'centroid_score_threshold': searcher.config.centroid_score_threshold
    # default: 'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(16 * np.sqrt(n_vec))))',
    # default: 'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            # 'dataset_l': ['lotte', 'msmacro'],
            # 'dataset_l': ['lotte-lifestyle', 'lotte', 'msmacro'],
            # 'dataset_l': ['quora'],
            'dataset_l': [
                {'dataset': 'hotpotqa', 'n_centroid': 262144, 'n_bit': 2, 'n_pid_kmeans': -1},
                {'dataset': 'lotte', 'n_centroid': 131072, 'n_bit': 2, 'n_pid_kmeans': 273157},
                {'dataset': 'msmacro', 'n_centroid': 262144, 'n_bit': 2, 'n_pid_kmeans': 519938},
                {'dataset': 'wiki-nq', 'n_centroid': 524288, 'n_bit': 2, 'n_pid_kmeans': -1},
            ],
            # 'topk_l': [10, 100, 1000],
            'topk_l': [100],
            'retrieval_parameter_l': [
                {"ndocs": 256, "ncells": 1, "centroid_score_threshold": 0.5, "n_thread": 1},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'ndocs': [4 * 10, 4 * 30, 4 * 60, 4 * 120, 4 * 250, 4 * 500, 4 * 1000],
                    'ncells': [1, 2],
                    'centroid_score_threshold': [0.5, 0.6, 0.7, 0.8],
                    'n_thread': [1]

                    # 'ndocs': [256],
                    # 'ncells': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    # 'centroid_score_threshold': [0.5],
                    # 'n_thread': [1]
                },
                100: {
                    # 'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400, 4 * 500, 4 * 600, 4 * 700, 4 * 800, 4 * 900, 4 * 1000],
                    'ndocs': [4 * 120, 4 * 250, 4 * 500, 4 * 1000, 4 * 1500, 4 * 2000],
                    'ncells': [1, 2, 3],
                    'centroid_score_threshold': [0.45, 0.6, 0.8],
                    'n_thread': [1]
                },
                1000: {
                    # 'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400, 4 * 500, 4 * 600, 4 * 700, 4 * 800, 4 * 900, 4 * 1000],
                    'ndocs': [4 * 1000, 4 * 1500, 4 * 2000, 4 * 3500],
                    'ncells': [1, 4, 8],
                    'centroid_score_threshold': [0.4],
                    'n_thread': [1]
                }
            }
        },
        'local': {
            'username': 'bianzheng',
            'dataset_l': [
                {'dataset': 'lotte-500-gnd', 'n_centroid': 2048, 'n_bit': 2, 'n_pid_kmeans': 500},
                # {'dataset': 'lotte-500-gnd', 'n_centroid': 1024, 'n_bit': 2, 'n_pid_kmeans': 500},
                # {'dataset': 'lotte-500-gnd', 'n_centroid': 512, 'n_bit': 2, 'n_pid_kmeans': 500}
            ],
            'topk_l': [10],
            'retrieval_parameter_l': [
                {'ndocs': 32, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                {'ndocs': 128, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                {'ndocs': 512, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                {'ndocs': 512, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": -1},
            ],
            'grid_search': False,
            'grid_search_para': {
                10: {
                    'ndocs': [4 * 10, 4 * 50, 4 * 100, 4 * 200],
                    'ncells': [1, 2],
                    'centroid_score_threshold': [0.5],
                    'n_thread': [1]
                },
                50: {
                    'ndocs': [4 * 50, 4 * 100, 4 * 200, 4 * 300],
                    'ncells': [1, 2],
                    'centroid_score_threshold': [0.5],
                    'n_thread': [1]
                }
            }
        }
    }
    host_name = 'local'

    config = config_l[host_name]
    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']
    search_gpu = True

    for dataset_info in dataset_l:
        dataset = dataset_info['dataset']
        n_centroid = dataset_info['n_centroid']
        n_bit = dataset_info['n_bit']
        n_pid_kmeans = dataset_info['n_pid_kmeans']
        for topk in topk_l:
            grid_search = config['grid_search']
            if grid_search:
                retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
            else:
                retrieval_parameter_l = config['retrieval_parameter_l']

            retrieval(username=username, dataset=dataset, retrieval_parameter_l=retrieval_parameter_l, topk=topk,
                      n_centroid=n_centroid, n_bit=n_bit, n_pid_kmeans=n_pid_kmeans, search_gpu=search_gpu)
