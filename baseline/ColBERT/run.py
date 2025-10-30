import json

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# export CUDA_VISIBLE_DEVICES=""
# in bash, setting CUDA_VISIBLE_DEVICES=1 to enable
# export CUDA_VISIBLE_DEVICES="0"
from os import listdir
from os.path import isfile, join
import re
from typing import Callable

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher, IndexerGenerate
from colbert.data import Queries
import numpy as np
import torch
import time
from memory_profiler import memory_usage
import sys
import copy
import pandas as pd

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def get_n_chunk(base_dir: str):
    filename_l = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)


def build_index_official(username: str, dataset: str, n_centroid_f: str, n_bit: int, sample_pid_f: str,
                         save_embedding: bool):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval-gpu/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    if save_embedding:
        os.makedirs(embedding_path, exist_ok=True)
        command = f'rm -rf {embedding_path}/*'
        print(command)
        os.system(command)
        # delete_file_if_exist(embedding_path)
        os.makedirs(base_embedding_path, exist_ok=False)
    else:
        assert os.path.exists(base_embedding_path)
    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'experiments'))):
        config = ColBERTConfig(
            nbits=n_bit,
            n_centroid_f=n_centroid_f,
            sample_pid_f=sample_pid_f,
            root=colbert_project_path,
        )
        indexer = Indexer(checkpoint=pretrain_index_path, config=config)
        build_index_time, encode_passage_time, n_partition, n_pid_kmeans = indexer.index(name=dataset,
                                                                                         collection=os.path.join(
                                                                                             document_data_path,
                                                                                             'collection.tsv'),
                                                                                         embedding_filename=base_embedding_path,
                                                                                         save_embedding=save_embedding,
                                                                                         overwrite=True)
    index_origin_path = os.path.join(colbert_project_path, f'experiments/{dataset}/indexes/{dataset}')
    index_new_path = os.path.join(index_path,
                                  f'plaid-n_centroid_{n_partition}-n_bit_{n_bit}-n_pid_kmeans_{n_pid_kmeans}')
    delete_file_if_exist(index_new_path)
    os.makedirs(index_new_path, exist_ok=False)
    os.system(f'mv {index_origin_path}/* {index_new_path}')
    print("finish indexing, start searching")

    build_index_json = {'build_index_time (s)': build_index_time, 'encode_passage_time (s)': encode_passage_time}
    with open(os.path.join(result_performance_path,
                           f'{dataset}-build_index-plaid-n_centroid_{n_partition}-n_bit_{n_bit}-n_pid_kmeans_{n_pid_kmeans}.json'),
              'w') as f:
        json.dump(build_index_json, f)

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        total_encode_time_ms, n_encode_query = searcher.save_query_embedding(queries, query_embedding_filename)
        # topk = 100

        # ranking = searcher.search_all_embedding(query_embedding_filename, k=topk)
        # ranking.save(f"{dataset}_self_search_method_top{topk}.tsv")

        # ranking = searcher.search_all(queries, k=topk)
        # ranking.save(f"{dataset}_official_search_method_top{topk}.tsv")

    encode_info = {'total_encode_time_ms': total_encode_time_ms, 'n_encode_query': n_encode_query,
                   'average_encode_time_ms': total_encode_time_ms / n_encode_query}
    with open(os.path.join(result_performance_path, f'{dataset}-encode_query.json'), 'w') as f:
        json.dump(encode_info, f)

    n_chunk = get_n_chunk(base_embedding_path)
    total_doclens = []
    for chunkID in range(n_chunk):
        doclens = np.load(os.path.join(base_embedding_path, f'doclens{chunkID}.npy'))
        total_doclens = np.append(total_doclens, doclens)
    np.save(os.path.join(embedding_path, 'doclens.npy'), total_doclens)
    return n_partition, n_pid_kmeans


def encode_query_cpu(username: str, dataset: str):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval-gpu/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')
    print("finish indexing, start searching")

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        total_encode_time_ms, n_encode_query = searcher.save_query_embedding_cpu(queries, query_embedding_filename)
        # topk = 100

        # ranking = searcher.search_all_embedding(query_embedding_filename, k=topk)
        # ranking.save(f"{dataset}_self_search_method_top{topk}.tsv")

        # ranking = searcher.search_all(queries, k=topk)
        # ranking.save(f"{dataset}_official_search_method_top{topk}.tsv")

    encode_info = {'total_encode_time_ms': total_encode_time_ms, 'n_encode_query': n_encode_query,
                   "cpu": True,
                   'average_encode_time_ms': total_encode_time_ms / n_encode_query}
    with open(os.path.join(result_performance_path, f'{dataset}-encode_query.json'), 'w') as f:
        json.dump(encode_info, f)


def build_index_generate(username: str, dataset: str):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval-gpu/baseline/ColBERT'
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    indexer = IndexerGenerate()
    build_index_time, encode_passage_time = indexer.index(username=username, dataset=dataset)

    build_index_json = {'build_index_time (s)': build_index_time, 'encode_passage_time (s)': encode_passage_time}
    with open(os.path.join(result_performance_path, f'{dataset}-build_index-plaid-.json'), 'w') as f:
        json.dump(build_index_json, f)


def load_training_query(username: str, dataset: str, n_sample_query: int):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval-gpu/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Index/{dataset}'

    n_gpu = torch.cuda.device_count()
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.train.tsv'))
        train_query = searcher.get_query_embedding(queries)
        del searcher, queries
    return train_query


def load_dev_query(username: str, dataset: str, n_sample_query: int):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval-gpu/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Index/{dataset}'

    n_gpu = torch.cuda.device_count()
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        train_query = searcher.get_query_embedding(queries)
        del searcher, queries
    return train_query


def retrieval_official(username: str, dataset: str, topk: int, search_config_l: list,
                       n_centroid: int, n_bit: int, n_pid_kmeans: int,
                       search_gpu: bool):
    colbert_project_path = f'/home/{username}/multi-vector-retrieval-gpu/baseline/ColBERT'
    raw_data_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/RawData'
    pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Index/{dataset}'
    result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/performance'
    result_answer_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/answer'
    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                topk=topk)

    build_index_suffix = f'n_centroid_{n_centroid}-n_bit_{n_bit}-n_pid_kmeans_{n_pid_kmeans}'
    suffix_index_filename = os.path.join(index_path, f'plaid-{build_index_suffix}')
    nonsuffix_index_filename = os.path.join(index_path, f'plaid')
    print(f'suffix_index_filename {suffix_index_filename}, nonsuffix_index_filename {nonsuffix_index_filename}, os.path.exists(suffix_index_filename) {os.path.exists(suffix_index_filename)}')
    index_new_path = suffix_index_filename if os.path.exists(suffix_index_filename) else nonsuffix_index_filename
    device_suffix = 'gpu' if search_gpu else 'cpu'
    module_name = f'plaid-{device_suffix}'

    query_emb = np.load(query_embedding_filename)
    n_query = len(query_emb)
    qid_l = []
    with open(query_text_filename, 'r') as f:
        for line in f:
            query_text_l = line.split('\t')
            qid_l.append(int(query_text_l[0]))
        assert len(qid_l) == n_query

    final_result_l = []
    with Run().context(
            RunConfig(nranks=n_gpu, search_gpu=search_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):

        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv'),
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)

        for search_config in search_config_l:
            print(f"plaid topk {topk}, search config {search_config}")
            if search_config['n_thread'] == -1:
                torch.set_num_threads(torch.get_num_threads())
            else:
                torch.set_num_threads(search_config['n_thread'])
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            searcher.configure(**colbert_retrieval_config)

            mem_usage, retrieve_res = \
                memory_usage((searcher.search_all_embedding_by_vector, (), {'query_emb': query_emb,
                                                                            'query_embd_filename': query_embedding_filename,
                                                                            'qid_l': qid_l, 'k': topk}), interval=0.3,
                             retval=True)
            ranking, retrieval_time_l, time_ivf_l, time_filter_l, time_refine_l, n_refine_ivf_l, n_refine_filter_l, n_vec_score_refine_l = retrieve_res

            time_ms_l = np.around(retrieval_time_l, 3)

            para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
            retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                               f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
            ranking.save_absolute_path(
                os.path.join(result_answer_path,
                             f'{dataset}-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'))

            search_time_m = {
                'total_query_time_ms': '{:.3f}'.format(sum(retrieval_time_l)),
                'peak_memory_usage_mb': '{:.3f}'.format(np.max(mem_usage)),
                "retrieval_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
                "retrieval_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
                "retrieval_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
                'average_query_time_ms': '{:.3f}'.format(1.0 * sum(retrieval_time_l) / n_query),
                'average_ivf_time_ms': '{:.3f}'.format(1.0 * sum(time_ivf_l) / n_query),
                'average_filter_time_ms': '{:.3f}'.format(1.0 * sum(time_filter_l) / n_query),
                'average_refine_time_ms': '{:.3f}'.format(1.0 * sum(time_refine_l) / n_query),
                'average_n_refine_ivf': '{:.3f}'.format(np.average(n_refine_ivf_l)),
                'average_n_refine_filter': '{:.3f}'.format(np.average(n_refine_filter_l)),
                'average_n_vec_score_refine': '{:.3f}'.format(np.average(n_vec_score_refine_l)),
            }
            retrieval_config = {
                'ndocs': searcher.config.ndocs,
                'ncells': searcher.config.ncells,
                'centroid_score_threshold': searcher.config.centroid_score_threshold,
                'n_thread': search_config['n_thread'],
                'search_device': device_suffix,
            }
            recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
                username=username, dataset=dataset, topk=topk,
                method_name=module_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
                mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)
            retrieval_info_m = {
                'n_query': n_query, 'topk': topk, 'build_index': {},
                'retrieval': retrieval_config,
                'search_time': search_time_m, 'search_accuracy': search_accuracy_m
            }

            method_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
            result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/performance'
            performance_filename = os.path.join(result_performance_path, method_performance_name)
            with open(performance_filename, "w") as f:
                json.dump(retrieval_info_m, f)

            df = pd.DataFrame({'time(ms)': time_ms_l, 'recall': recall_l})
            df.index.name = 'local_queryID'
            if mrr_l:
                df['mrr'] = mrr_l
            if success_l:
                df['success'] = success_l
            single_query_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.csv'
            result_performance_path = f'/home/{username}/Dataset/multi-vector-retrieval-gpu/Result/single_query_performance'
            single_query_performance_filename = os.path.join(result_performance_path, single_query_performance_name)
            df.to_csv(single_query_performance_filename, index=True)

            print("#############final result###############")
            print("filename", method_performance_name)
            print("search time", retrieval_info_m['search_time'])
            print("search accuracy", retrieval_info_m['search_accuracy'])
            print("########################################")

            final_result_l.append({'filename': method_performance_name, 'search_time': retrieval_info_m['search_time'],
                                   'search_accuracy': retrieval_info_m['search_accuracy']})

        for final_result in final_result_l:
            print("#############final result###############")
            print("filename", final_result['filename'])
            print("search time", final_result['search_time'])
            print("search accuracy", final_result['search_accuracy'])
            print("########################################")


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-small'
    build_index_official(username=username, dataset=dataset)
