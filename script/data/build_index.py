import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
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
    # default settings: 'n_centroid_f': lambda n_vec: int(2 ** np.floor(np.log2(16 * np.sqrt(n_vec))))
    # default settings: 'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'
    config_l = {
        'dbg': {
            'username': 'zhengbian',
            'topk_l': [10, 100, 1000],
            'dataset_info_l': [
                # 'lotte',
                # 'msmacro',
                # 'wikipedia',
                # 'lotte-lifestyle',
                # 'quora',
                # 'hotpotqa',
                # 'wiki-nq',
                {'dataset': 'lotte',
                 'para_l': [
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(16 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(8 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(4 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(2 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(1 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(1 / 2 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(1 / 4 * np.sqrt(n_vec))))',
                      'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                 ]},
            ]
        },
        'local': {
            'username': 'bianzheng',
            'topk_l': [10, 50],
            'dataset_info_l': [
                {'dataset': 'lotte-500-gnd',
                 'para_l': [
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(16 * np.sqrt(n_vec))))', 'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 16 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(8 * np.sqrt(n_vec))))', 'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 8 * np.sqrt(typical_doclen * num_passages)'},
                     {'n_centroid_f': 'lambda n_vec: int(2 ** np.floor(np.log2(4 * np.sqrt(n_vec))))', 'n_bit': 2,
                      'sample_pid_f': 'lambda typical_doclen, num_passages: 4 * np.sqrt(typical_doclen * num_passages)'},
                 ]},
            ]
        }
    }
    host_name = 'local'
    config = config_l[host_name]
    username = config['username']
    topk_l = config['topk_l']
    dataset_info_l = config['dataset_info_l']

    for dataset_info in dataset_info_l:
        dataset = dataset_info['dataset']
        para_l = dataset_info['para_l']
        first_save_embedding = True
        for para in para_l:
            n_centroid_f = para['n_centroid_f']
            n_bit = para['n_bit']
            sample_pid_f = para['sample_pid_f']
            print(bcolors.OKGREEN + f"plaid start {dataset}" + bcolors.ENDC)
            if first_save_embedding:
                save_embedding = True
                first_save_embedding = False
            else:
                save_embedding = False
            print(
                bcolors.OKGREEN + f"n_bit {n_bit} n_centroid_f {n_centroid_f} save_embedding {save_embedding}" + bcolors.ENDC)
            colbert_run.build_index_official(username=username, dataset=dataset,
                                             n_centroid_f=n_centroid_f, n_bit=n_bit,
                                             sample_pid_f=sample_pid_f,
                                             save_embedding=save_embedding)
            print(bcolors.OKGREEN + f"plaid finish {dataset}" + bcolors.ENDC)

        module_name = 'BruteForceProgressive'
        print(bcolors.OKGREEN + f"groundtruth start {dataset}" + bcolors.ENDC)
        util.compile_file(username=username, module_name=module_name, is_debug=True)
        est_dist_l_l, est_id_l_l = groundtruth.gnd_cpp(username=username, dataset=dataset, topk_l=topk_l,
                                                       compile_file=False, module_name=module_name)
        for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
            groundtruth.save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset,
                                     topk=topk)
        print(bcolors.OKGREEN + f"groundtruth end {dataset}" + bcolors.ENDC)
