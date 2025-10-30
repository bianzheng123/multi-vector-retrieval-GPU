# Efficient GPU-based Multi-Vector Retrieval 
## Introduction
This is a fast GPU-based algorithm for the multi-vector retrieval problem based on ColBERT.

## Requirement (includes but not limited)

- Linux OS (Ubuntu-20.04)
- pybind, eigen, spdlog, CUDA 
- some important libraries 

## How to build and run

The experimental python version is 3.8

1. download the ColBERT pretrain model in https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
2. decompress the ColBERT model and move it into the file `{project_file}/multi-vector-retrieval-data/RawData/colbert-pretrain`
3. `mv {project_file}/multi-vector-retrieval-data /home/{username}/Dataset/multi-vector-retrieval-GPU`
4. set username in the following scripts described

Note that you should move the project path as `/home/{username}/`

Install the necessary library when the system reports an error

## Important scripts

- `script/data/build_index.py`, build the index of plaid, it also generates the embedding. The embeddings are used to build the index of GIGP. 
- `script/data/build_index_by_sample.py`, if you want to test the program in a small dataset, use this one
- `script/evaluation/eval_igp.py`, 
- `script/evaluation/eval_igp_gpu_basic.py`, the basic solution 
- `script/evaluation/eval_igp_gpu_pp.py`, the proposed advanced solution 
- `script/evaluation/eval_plaid_cpu.py`, cpu version of PLAID 
- `script/evaluation/eval_plaid_gpu.py`, gpu version of PLAID

## Dataset information

See `dataset.md`

## Prebuilt index

We have prepared the prebuild index of the evaluated dataset, i.e., LoTTE pooled, and MS MARCO, 

Please refer to https://connectpolyu-my.sharepoint.com/:f:/g/personal/21041743r_connect_polyu_hk/ElyiRI2Y0FNPmoP0ecApAUwB3v3gL_aNSHZaIrifCV_jDA

## Some tips for library installation:

1. pybind: build file, then make install

2. eigen: sudo apt install libeigen3-dev

3. to install spdlog, you should install the spdlog-1.11.0
when build and install, run `cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && make -j` to build the spdlog

4. when encountering the error: no CUDA runtime found, using CUDA_HOME = 'xxxx'

It may because of the incompatible version between CUDA and pytorch, the correct answer is to reinstall the pytorch and not do anything for the CUDA
A command that has worked before:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

5. When using the colbert to connect to huggingface. If it report the error show that huggingface cannot connect, then please add the proxy for downloading.
The detail is to add the proxy configuration at the line 'loadedConfig  = AutoConfig.from_pretrained(name_or_path)' in ColBERT/colbert/modeling/hf_colbert.py
You can learn how to use the proxychains for climbing the "wall"

6. For Colbert, when want to use GPU for build index / retrieval, set total_visible_gpus in ColBERT/colbert/infra/config/settings, the total_visible_gpus variable

cmake -DCMAKE_BUILD_TYPE=Debug ..

7. if you find the error `AttributeError: module 'faiss' has no attribute 'StandardGpuResources'`, or find the error about the matrix multiplication for calling the openblas, then you need to uninstall faiss-gpu in pip version and install the faiss-gpu using conda command, `pip install faiss-gpu-cu12`
