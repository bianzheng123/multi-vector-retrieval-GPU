# Dataset


| Dataset   | No. Document | No. Query |
|-----------|------------|-----------|
| Lotte     | 2.4M       | 2,931     |
| HotpotQA  | 5.2M       | 7,405     |
| MS MARCO  | 8.8M       | 6,980     |
| Wikipedia | 21.0M      | 6,515     |



### Lotte

similar as Lotte-lifestyle

Put the raw dataset in /home/xxx/Dataset/multi-vector-retrieval-gpu/RawData/lotte

https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz

File needed to be transformed manually:

Note that the base directory is /home/xxx/Dataset/multi-vector-retrieval-gpu/RawData/lotte

| Original file                          | Transformed file           |
| -------------------------------------- | -------------------------- |
| lotte/pooled/dev/collection.tsv        | document/collection.tsv    |
| lotte/pooled/dev/questions.search.tsv  | document/queries.dev.tsv   |
| lotte/pooled/test/questions.search.tsv | document/queries.train.tsv |

Then run `python3 script/preprocess_dataset/lotte.py` to generate the groundtruth file

### HotpotQA

downloaded from https://huggingface.co/datasets/BeIR/beir

link https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip

use `qrels/test.tsv` as the groundtruth

run `python3 script/preprocess_dataset/hotpotqa.py` for processing

### MS MARCO 

https://microsoft.github.io/msmarco/Datasets.html
you should download:  collectionandqueries.tar.gz 2.9GB
use: collection.tsv, qrels.dev.small.tsv as the groundtruth and queries.dev.small.tsv as the testing query
top-k of queries.train.tsv -> queries.train.tsv


### Wikipedia

https://ir-datasets.com/dpr-w100.html#dpr-w100/natural-questions/dev

use `python3 script/preprocess_dataset/download_ir_datasets.py` for download.


