import json
import numpy as np
import os


def extract_gnd_lotte(username: str, dataset: str):
    raw_data_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData'
    gnd_filename = os.path.join(raw_data_path, f'{dataset}/raw_data/pooled/dev/qas.search.jsonl')
    old_gnd_l = []
    with open(gnd_filename, 'r') as f:
        for line in f:
            old_gnd_l.append(json.loads(line))
    new_gnd_l = []
    for old_gnd_json in old_gnd_l:
        new_tmp_gnd = {
            'query_id': old_gnd_json['qid'],
            'passage_id': old_gnd_json['answer_pids'],
        }
        new_gnd_l.append(new_tmp_gnd)
    return new_gnd_l


def generate_gnd(username: str, dataset: str):
    raw_data_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData/'
    preprocess_gnd_filename = f'{raw_data_path}/{dataset}/document/queries.gnd.jsonl'

    gnd_l = extract_gnd_lotte(username=username, dataset=dataset)

    with open(preprocess_gnd_filename, 'w') as f:
        for gnd_json in gnd_l:
            f.write(json.dumps(gnd_json) + '\n')


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte'
    generate_gnd(username=username, dataset=dataset)
