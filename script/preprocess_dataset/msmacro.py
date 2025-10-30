import json
import numpy as np


def generate_gnd(username: str):
    raw_data_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData/'
    raw_query_filename = f'{raw_data_path}/msmacro/raw_data/queries.dev.small.tsv'
    raw_gnd_filename = f'{raw_data_path}/msmacro/raw_data/qrels.dev.small.tsv'

    preprocess_query_filename = f'{raw_data_path}/msmacro/document/queries.dev.tsv'
    preprocess_gnd_filename = f'{raw_data_path}/msmacro/document/queries.gnd.jsonl'

    query_text_m = {}
    with open(raw_query_filename, 'r') as f:
        for line in f:
            line_seg_l = line.split('\t')
            queryID = int(line_seg_l[0])
            query_text = line_seg_l[1]
            query_text_m[queryID] = query_text
    print(f'finish gen raw_query, len {len(query_text_m.keys())}')

    query_passage_m = {}
    with open(raw_gnd_filename, 'r') as f:
        for line in f:
            txt_l = line.split('\t')
            queryID = int(txt_l[0])
            queryID_symbol = int(txt_l[1])
            passageID = int(txt_l[2])
            passage_symbol = int(txt_l[3])
            if queryID not in query_passage_m:
                query_passage_m[queryID] = [passageID]
            else:
                query_passage_m[queryID].append(passageID)
    print(f'finish gen queries with passage answer, len {len(query_passage_m.keys())}')

    queryID_l = np.sort(list(query_passage_m.keys())).tolist()
    with open(preprocess_query_filename, 'w') as f:
        for queryID in queryID_l:
            query_text = query_text_m[queryID]
            f.write(f'{queryID}\t{query_text}')

    with open(preprocess_gnd_filename, 'w') as f:
        for queryID in queryID_l:
            passageID_l = np.sort(query_passage_m[queryID]).tolist()
            gnd_json = {'query_id': queryID, 'passage_id': passageID_l}
            f.write(json.dumps(gnd_json) + '\n')


if __name__ == '__main__':
    username = 'bianzheng'
    generate_gnd(username=username)
