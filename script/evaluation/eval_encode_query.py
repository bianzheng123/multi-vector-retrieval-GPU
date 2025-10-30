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
from baseline.ColBERT import run as colbert_run

if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'
    colbert_run.encode_query_cpu(username, dataset)