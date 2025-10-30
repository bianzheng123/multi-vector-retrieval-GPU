import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'Dessert')
sys.path.append(ROOT_PATH)

from baseline.ColBERT import run as colbert_run
from script.data import groundtruth
import util


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    username = 'bianzheng'
    topk_l = [10, 100]
    dataset_l = [
        'deep-fake',
    ]
    for dataset in dataset_l:
        print(bcolors.OKGREEN + f"plaid start {dataset}" + bcolors.ENDC)
        colbert_run.build_index_generate(username=username, dataset=dataset)
        print(bcolors.OKGREEN + f"plaid finish {dataset}" + bcolors.ENDC)
