import json
import numpy as np
import os
import jsonlines


def process_document(username: str, dataset: str):
    raw_data_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData/'
    raw_document_fname = os.path.join(raw_data_path, f'{dataset}/hotpotqa/corpus.jsonl')
    old_doc_l = []
    with open(raw_document_fname, 'r') as f:
        for line in f:
            j_ins = json.loads(line)
            old_doc_l.append({'id': j_ins['_id'], 'text': j_ins['text']})

    document_fname = os.path.join(raw_data_path, f'{dataset}/document/collection.tsv')
    with open(document_fname, 'w') as f:
        for doc in old_doc_l:
            f.write(f'{doc["id"]}\t{doc["text"]}\n')


def read_qrel(username: str, dataset: str, qrel_fname: str):
    # process the training query, read dev.tsv and extract the training query
    qrel_m = {}  # map of {queryID_str, [docID]}
    raw_data_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData/'
    train_qrel_fname = os.path.join(raw_data_path, f'{dataset}/hotpotqa/qrels/{qrel_fname}')
    with open(train_qrel_fname, 'r') as f:
        txt_l = f.read().split('\n')
        print(f"n_qrel = {len(txt_l) - 1}")
        for i in range(1, len(txt_l), 1):
            if len(txt_l[i].split('\t')) == 1:
                continue
            queryID_str, docID, score = txt_l[i].split('\t')
            docID = int(docID)
            if queryID_str in qrel_m.keys():
                qrel_m[queryID_str].append(docID)
            else:
                qrel_m[queryID_str] = [docID]
    return qrel_m


def process_query_gnd(username: str, dataset: str):
    raw_data_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData/'
    # first rename for each query, because the queryID is a string
    queryID_rename_m = {}  # map of {queryID_str, queryID_int}
    query_m = {}  # map of {queryID_str, text}
    query_answer_m = {} # map of {queryID_str, answer}
    query_jsonl_fname = os.path.join(raw_data_path, f'{dataset}/hotpotqa/queries.jsonl')
    rename_queryID = 1
    with open(query_jsonl_fname, 'r') as f:
        for line in f:
            j_ins = json.loads(line)
            queryID_str = j_ins['_id']
            assert queryID_str not in queryID_rename_m
            queryID_rename_m[queryID_str] = rename_queryID
            query_answer_m[queryID_str] = j_ins['metadata']['answer']
            rename_queryID += 1

            text = j_ins['text']
            query_m[queryID_str] = text

    # process the training query, read dev.tsv and extract the training query
    train_qrel_m = read_qrel(username=username, dataset=dataset, qrel_fname='dev.tsv')  # map of {queryID_str, [docID]}
    train_query_l = []  # list of [queryID_int, text]
    for queryID_str in train_qrel_m.keys():
        queryID_int = queryID_rename_m[queryID_str]
        text = query_m[queryID_str]
        train_query_l.append([queryID_int, text])

    train_query_fname = os.path.join(raw_data_path, f'{dataset}/document/queries.train.tsv')
    with open(train_query_fname, 'w') as f:
        for train_query in train_query_l:
            transformed_queryID, text = train_query
            f.write(f'{transformed_queryID}\t{text}\n')

    # process the testing query, read test.tsv and extract the testing query and qrel
    test_qrel_m = read_qrel(username=username, dataset=dataset, qrel_fname='test.tsv')  # map of {queryID_str, [docID]}
    test_query_l = []  # list of [queryID_int, text]
    test_query_answer_l = [] # map of {"query_id": queryID_int, "answers": list of answer}
    for queryID_str in test_qrel_m.keys():
        queryID_int = queryID_rename_m[queryID_str]
        text = query_m[queryID_str]
        test_query_l.append([queryID_int, text])
        test_query_answer_l.append({"query_id": queryID_int, 'answers': [query_answer_m[queryID_str]]})

    test_query_fname = os.path.join(raw_data_path, f'{dataset}/document/queries.dev.tsv')
    with open(test_query_fname, 'w') as f:
        for test_query in test_query_l:
            transformed_queryID, text = test_query
            f.write(f'{transformed_queryID}\t{text}\n')

    test_qrel_int_l = []  # list of [queryID_int, [docID]]
    for queryID_str in test_qrel_m.keys():
        queryID_int = queryID_rename_m[queryID_str]
        docID_l = test_qrel_m[queryID_str]
        test_qrel_int_l.append([queryID_int, docID_l])

    preprocess_gnd_filename = f'{raw_data_path}/{dataset}/document/queries.gnd.jsonl'
    with open(preprocess_gnd_filename, 'w') as f:
        for queryID_int, docID_l in test_qrel_int_l:
            j_ins = {"query_id": queryID_int, "passage_id": docID_l}
            f.write(json.dumps(j_ins) + '\n')

    show_answer_gnd_fname = f'{raw_data_path}/{dataset}/document/queries_short_answer.gnd.jsonl'
    with jsonlines.open(show_answer_gnd_fname, 'w') as writer:
        writer.write_all(test_query_answer_l)
    print(f"complete the hotpotqa ground truth write file, dataset {dataset}")


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'hotpotqa'
    process_document(username=username, dataset=dataset)
    process_query_gnd(username=username, dataset=dataset)
