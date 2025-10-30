import os
import ir_datasets
import json


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def download(ir_dataset: str):
    ir_dataset = ir_datasets.load(ir_dataset)
    # print(ir_dataset.docs_count())
    # print(ir_dataset.docs_cls())
    doc_l = []
    for doc in ir_dataset.docs_iter():
        doc_l.append(doc)
    print(f"len(doc_l) {len(doc_l)}")
    query_l = []
    for query in ir_dataset.queries_iter():
        # print(query, query.query_id, query.text)
        query_l.append(query)
    print(f"len(query_l) {len(query_l)}")
    qrel_l = []
    for qrel in ir_dataset.qrels_iter():
        # print(qrel, qrel.query_id, qrel.doc_id, qrel.relevance)
        qrel_l.append(qrel)
    print(f"len(qrel_l) {len(qrel_l)}")


def save(ir_dataset: str, dataset_name: str, username: str):
    rawdata_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/RawData/{dataset_name}'
    rawdata_doc_path = os.path.join(rawdata_path, 'document')
    delete_file_if_exist(rawdata_doc_path)
    os.makedirs(rawdata_doc_path, exist_ok=False)
    ir_dataset = ir_datasets.load(ir_dataset)

    docID_raw2count_m = {}
    count = 0
    for doc in ir_dataset.docs_iter():
        docID_raw2count_m[doc.doc_id] = count
        count += 1
    with open(os.path.join(rawdata_path, 'docID_raw2count_m.json'), 'w') as f:
        json.dump(docID_raw2count_m, f)

    with open(os.path.join(rawdata_doc_path, 'collection.tsv'), 'w') as f:
        for doc in ir_dataset.docs_iter():
            docID_count = docID_raw2count_m[doc.doc_id]
            f.write(f'{docID_count}\t{doc.text}\n')
    print(f"len(doc_l) {count}")


    queryID_raw2count_m = {}
    count = 0
    for query in ir_dataset.queries_iter():
        queryID_raw2count_m[query.query_id] = count
        count += 1
    with open(os.path.join(rawdata_path, 'queryID_raw2count_m.json'), 'w') as f:
        json.dump(queryID_raw2count_m, f)

    with open(os.path.join(rawdata_doc_path, 'queries.dev.tsv'), 'w') as f:
        for query in ir_dataset.queries_iter():
            queryID_count = queryID_raw2count_m[query.query_id]
            f.write(f'{queryID_count}\t{query.text}\n')
    print(f"len(query_l) {count}")


    qrel_doc2query_m = {}
    for qrel in ir_dataset.qrels_iter():
        docID_count = docID_raw2count_m[qrel.doc_id]
        queryID_count = queryID_raw2count_m[qrel.query_id]
        if qrel.relevance > 0:
            if queryID_count not in qrel_doc2query_m:
                qrel_doc2query_m[queryID_count] = [docID_count]
            else:
                qrel_doc2query_m[queryID_count].append(docID_count)

    with open(os.path.join(rawdata_doc_path, 'queries.gnd.jsonl'), 'w') as f:
        for query in ir_dataset.queries_iter():
            queryID_count = queryID_raw2count_m[query.query_id]
            docID_count_l = qrel_doc2query_m[queryID_count]
            gnd_json = {
                "query_id": queryID_count,
                "passage_id": docID_count_l,
            }
            f.write(json.dumps(gnd_json) + '\n')


if __name__ == '__main__':
    username = 'zhengbian'
    ir_dataset = 'dpr-w100/natural-questions/dev'
    dataset_name = 'dpr-nq'
    print(f"ir_dataset {ir_dataset}, dataset_name {dataset_name}")
    download(ir_dataset=ir_dataset)
