//
// Created by Administrator on 7/8/2025.
//

#ifndef RERANK_HPP
#define RERANK_HPP

#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <raft/linalg/reduce.cuh>

namespace VectorSetSearch {

__global__ void vecID2vecOffset(const uint32_t *rerank_item_n_vec_l,
                                const uint32_t *rerank_item_n_vec_offset_l,
                                const uint32_t max_item_n_vec,
                                const uint32_t n_thread_per_item,
                                const uint32_t probe_topk,
                                const uint32_t n_rerank_vec,
                                VecOffset *vecID2vec_offset_l) {
  // global_threadID = probe_topk * n_thread_per_item
  const uint32_t global_threadID = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t probeID = global_threadID / n_thread_per_item;
  if (probeID >= probe_topk) return;
  const uint32_t item_threadID = global_threadID % n_thread_per_item;
  const uint32_t item_n_vec = rerank_item_n_vec_l[probeID];
  assert(item_n_vec <= max_item_n_vec);
  const uint32_t offset = rerank_item_n_vec_offset_l[probeID];

  for (uint32_t item_vecID = item_threadID; item_vecID < item_n_vec; item_vecID += n_thread_per_item) {
    vecID2vec_offset_l[offset + item_vecID].item_offset_ = offset;
    vecID2vec_offset_l[offset + item_vecID].item_vecID_ = item_vecID;
    vecID2vec_offset_l[offset + item_vecID].item_n_vec_ = item_n_vec;
    vecID2vec_offset_l[offset + item_vecID].probeID_ = probeID;
    assert(offset + item_vecID < n_rerank_vec);
  }
}

__global__ void decompressVQCode(const VecOffset *vecID2vec_offset_l,
                                 const uint32_t *vq_code_l,
                                 const float *centroid_l,
                                 const uint32_t n_rerank_vec, const uint32_t n_thread_per_vec,
                                 const uint32_t vec_dim,
                                 const uint32_t max_item_n_vec,
                                 float *cand_vec_l) {
  // threadID = n_rerank_vec * n_thread_per_vec
  const uint32_t threadID = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= n_rerank_vec * n_thread_per_vec) return;
  const uint32_t stack_vecID = threadID / n_thread_per_vec;
  assert(stack_vecID < n_rerank_vec);
  const uint32_t vec_threadID = threadID % n_thread_per_vec;
  assert(vec_threadID < n_thread_per_vec);
  assert(vecID2vec_offset_l[stack_vecID].item_offset_ != (uint32_t)(-1));
  assert(vecID2vec_offset_l[stack_vecID].item_vecID_ != (uint32_t)(-1));
  assert(vecID2vec_offset_l[stack_vecID].item_n_vec_ != (uint32_t)(-1));
  const uint32_t item_offset = vecID2vec_offset_l[stack_vecID].item_offset_;
  const uint32_t item_vecID = vecID2vec_offset_l[stack_vecID].item_vecID_;
  const uint32_t probeID = vecID2vec_offset_l[stack_vecID].probeID_;
  assert(item_offset + item_vecID < n_rerank_vec);
  const uint32_t vq_code = vq_code_l[item_offset + item_vecID];
  const float *centroid = centroid_l + vq_code * vec_dim;
  float *cand_vec = cand_vec_l + probeID * max_item_n_vec * vec_dim + item_vecID * vec_dim;
  for (uint32_t dim = vec_threadID; dim < vec_dim; dim += n_thread_per_vec) {
    cand_vec[dim] = centroid[dim];
  }
}

__global__ void addSQCode(const VecOffset *vecID2vec_offset_l,
                          const uint8_t *sq_code_l,
                          const float *pack_code2weight_l, // pack_code_range_ * n_val_per_byte_
                          const uint32_t n_packed_val_per_vec,
                          const uint32_t pack_code_range, const uint32_t n_val_per_byte,
                          const uint32_t n_rerank_vec, const uint32_t n_thread_per_vec,
                          const uint32_t vec_dim,
                          const uint32_t max_item_n_vec,
                          float *cand_vec_l) {
  // load into the shared memory
  extern __shared__ float pack_code2weight_l_shared[];
  for (uint32_t i = threadIdx.x; i < pack_code_range * n_val_per_byte; i += blockDim.x) {
    pack_code2weight_l_shared[i] = pack_code2weight_l[i];
  }
  __syncthreads();

  // threadID = n_rerank_vec * n_thread_per_vec
  const uint32_t threadID = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= n_rerank_vec * n_thread_per_vec) return;
  const uint32_t stack_vecID = threadID / n_thread_per_vec;
  const uint32_t vec_threadID = threadID % n_thread_per_vec;
  const uint32_t item_offset = vecID2vec_offset_l[stack_vecID].item_offset_;
  const uint32_t item_vecID = vecID2vec_offset_l[stack_vecID].item_vecID_;
  const uint32_t item_n_vec = vecID2vec_offset_l[stack_vecID].item_n_vec_;
  const uint32_t probeID = vecID2vec_offset_l[stack_vecID].probeID_;
  assert(item_offset + item_vecID < n_rerank_vec);

  const uint8_t *residual_code = sq_code_l + item_offset * n_packed_val_per_vec +
      item_vecID * n_packed_val_per_vec;
  float *residual_vec = cand_vec_l + probeID * max_item_n_vec * vec_dim + item_vecID * vec_dim;

  for (uint32_t packed_codeID = vec_threadID; packed_codeID < n_packed_val_per_vec;
       packed_codeID += n_thread_per_vec) {
    const uint32_t start_dim = packed_codeID * n_val_per_byte;
    const uint8_t packed_code = residual_code[packed_codeID];
    const float *pack_code2weight = pack_code2weight_l_shared + packed_code * n_val_per_byte;
    assert(packed_code < pack_code_range);
    for (uint32_t valID = 0; valID < n_val_per_byte; valID++) {
      const uint32_t dim = start_dim + valID;
      const float weight = pack_code2weight[valID];
      assert(dim < vec_dim);
      residual_vec[dim] += weight;
    }
  }
}

class Rerank {
 public:
  ItemVecInfo item_vec_info_;
  VQInfo vq_info_;
  SQInfo sq_info_;
  const GPUResource *gpu_resource_;

  uint32_t query_n_vec_, probe_topk_;

  TimeRecordCUDA transmit_record_, compute_record_;

  std::vector<uint32_t> itemID_l_; // probe_topk
  std::vector<uint32_t> item_n_vec_l_; // probe_topk

  // transmit by cpu:
  uint32_t* vq_code_l_gpu_; // probe_topk * max_item_n_vec, stack storage
  uint8_t* sq_code_l_gpu_; // probe_topk * max_item_n_vec * n_packed_val_per_vec, stack storage
  uint32_t *item_n_vec_l_gpu_; // probe_topk
  // computed by GPU
  uint32_t *item_n_vec_offset_l_gpu_; // probe_topk
  VecOffset *vecID2vec_offset_l_gpu_; // probe_topk * max_item_n_vec
  float *vec_l_gpu_; // probe_topk * max_item_n_vec * vec_dim, pad storage
  float *score_table_gpu_; // query_n_vec * probe_topk * max_item_n_vec
  float *max_score_l_gpu_; // query_n_vec * probe_topk
  float *score_l_gpu_; // probe_topk

  Rerank() = default;

  Rerank(const ItemVecInfo &item_vec_info, const VQInfo &vq_info,
         const SQInfo &sq_info, const GPUResource *gpu_resource,
         const uint32_t query_n_vec, const uint32_t probe_topk) {
    this->item_vec_info_ = item_vec_info;
    this->vq_info_ = vq_info;
    this->sq_info_ = sq_info;
    this->gpu_resource_ = gpu_resource;

    this->query_n_vec_ = query_n_vec;
    this->probe_topk_ = probe_topk;

    itemID_l_.resize(probe_topk_);
    item_n_vec_l_.resize(probe_topk_);

    CHECK(cudaMalloc(&vq_code_l_gpu_, probe_topk_ * item_vec_info_.max_item_n_vec_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&sq_code_l_gpu_, probe_topk_ * item_vec_info_.max_item_n_vec_ *
                      sq_info_.n_packed_val_per_vec_ * sizeof(uint8_t)));
    CHECK(cudaMalloc(&item_n_vec_l_gpu_, probe_topk_ * sizeof(uint32_t)));

    CHECK(cudaMalloc(&item_n_vec_offset_l_gpu_, probe_topk_ * sizeof(uint32_t)));
    CHECK(cudaMalloc(&vecID2vec_offset_l_gpu_, probe_topk_ * item_vec_info_.max_item_n_vec_ * sizeof(VecOffset)));
    CHECK(cudaMalloc(&vec_l_gpu_,
                     probe_topk_ * item_vec_info_.max_item_n_vec_ * item_vec_info_.vec_dim_ * sizeof(float)));
    CHECK(cudaMalloc(&score_table_gpu_,
                     query_n_vec_ * probe_topk_ * item_vec_info_.max_item_n_vec_ * sizeof(float)));
    CHECK(cudaMalloc(&max_score_l_gpu_, query_n_vec_ * probe_topk_ * sizeof(float)));
    CHECK(cudaMalloc(&score_l_gpu_, probe_topk_ * sizeof(float)));
  }

  void reset() {
    CHECK(cudaMemset(vec_l_gpu_, 0,
               probe_topk_ * item_vec_info_.max_item_n_vec_ * item_vec_info_.vec_dim_ * sizeof(float)));
    CHECK(cudaMemset(score_table_gpu_, 0,
               query_n_vec_ * probe_topk_ * item_vec_info_.max_item_n_vec_ * sizeof(float)));
  }

  // output is score_l_gpu_
  void rerank(const float *query_gpu,
              const uint32_t *filter_itemID_l_gpu,
              double& rerank_transmit_time, double& rerank_compute_time,
              const uint32_t queryID) {
    nvtxRangePushA("OverallTransmitTime");
    CHECK(cudaMemcpy(itemID_l_.data(), filter_itemID_l_gpu,
                     probe_topk_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    nvtxRangePushA("rerank-prepareRerankCPU");
    uint32_t n_rerank_vec;
    transmitSQCode(n_rerank_vec, rerank_transmit_time);
    nvtxRangePop();
    nvtxRangePop();
    // in cpu, gather the item_n_vec, the offset, and the residual code

    RerankFunc(query_gpu, n_rerank_vec, rerank_compute_time);
  }

  void transmitSQCode(uint32_t &n_rerank_vec, double& transmit_time) {
    transmit_record_.start_record();
    n_rerank_vec = 0;
    for (uint32_t candID = 0; candID < probe_topk_; candID++)
    {
      const uint32_t itemID = itemID_l_[candID];
      assert(itemID < item_vec_info_.n_item_);
      const uint32_t item_n_vec = item_vec_info_.item_n_vec_l_[itemID];
      item_n_vec_l_[candID] = item_n_vec;

      const size_t item_offset = item_vec_info_.item_n_vec_offset_l_[itemID];
      const uint32_t* item_vq_code = vq_info_.vq_code_l_ + item_offset;
      CHECK(cudaMemcpyAsync(vq_code_l_gpu_ + n_rerank_vec, item_vq_code,
          item_n_vec * sizeof(uint32_t),
          cudaMemcpyHostToDevice));

      const uint8_t* item_residual_code = sq_info_.residual_code_l_ +
        item_offset * sq_info_.n_packed_val_per_vec_;
      CHECK(cudaMemcpyAsync(sq_code_l_gpu_ +
          (size_t) n_rerank_vec * sq_info_.n_packed_val_per_vec_,
          item_residual_code,
          sizeof(uint8_t) * item_n_vec * sq_info_.n_packed_val_per_vec_,
          cudaMemcpyHostToDevice));
      n_rerank_vec += item_n_vec;
    }

    CHECK(cudaMemcpy(item_n_vec_l_gpu_, item_n_vec_l_.data(),
        probe_topk_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpyAsync(rerank_item_n_vec_l_gpu_, rerank_item_n_vec_l_.data(),
    //     probe_topk_ * sizeof(uint32_t), cudaMemcpyHostToDevice,
    //     gpu_resource_->stream_l_[0]));
    CHECK(cudaDeviceSynchronize());
    transmit_time = transmit_record_.get_time_second();
  }

  // all of them are done in GPU
  void RerankFunc(const float *query_gpu,
                  const uint32_t n_rerank_vec,
                  double& compute_time) {
    compute_record_.start_record();
    nvtxRangePushA("rerank-decompressVQCode");
    // compute the offset of item_n_vec
    thrust::exclusive_scan(thrust::device, item_n_vec_l_gpu_, item_n_vec_l_gpu_ + probe_topk_,
                           item_n_vec_offset_l_gpu_);
    CHECK(cudaDeviceSynchronize());

    const uint32_t n_thread_per_item = 32;
    uint32_t block_size = 1024;
    uint32_t grid_size = (probe_topk_ * n_thread_per_item + block_size - 1) / block_size;
    vecID2vecOffset<<<grid_size, block_size>>>(
        item_n_vec_l_gpu_, item_n_vec_offset_l_gpu_,
        item_vec_info_.max_item_n_vec_, n_thread_per_item, probe_topk_, n_rerank_vec,
        vecID2vec_offset_l_gpu_);
    CHECK(cudaDeviceSynchronize());

    // decompress vq code
    uint32_t n_thread_per_vec = 32;
    block_size = 1024;
    grid_size = (n_rerank_vec * n_thread_per_vec + block_size - 1) / block_size;
    assert(n_rerank_vec < probe_topk_ * item_vec_info_.max_item_n_vec_);
    decompressVQCode<<<grid_size, block_size>>>(vecID2vec_offset_l_gpu_, vq_code_l_gpu_,
                                                vq_info_.centroid_l_gpu_,
                                                n_rerank_vec, n_thread_per_vec,
                                                item_vec_info_.vec_dim_,
                                                item_vec_info_.max_item_n_vec_,
                                                vec_l_gpu_);
    CHECK(cudaDeviceSynchronize());

    // add sq code
    block_size = 1024;
    n_thread_per_vec = 16;
    grid_size = (n_rerank_vec * n_thread_per_vec + block_size - 1) / block_size;
    assert(n_rerank_vec < probe_topk_ * item_vec_info_.max_item_n_vec_);
    addSQCode<<<grid_size, block_size,
    VectorSetSearch::SQInfo::pack_code_range_ * sq_info_.n_val_per_byte_ * sizeof(float)>>>(
        vecID2vec_offset_l_gpu_, sq_code_l_gpu_,
        sq_info_.pack_code2weight_l_gpu_,
        sq_info_.n_packed_val_per_vec_,
        VectorSetSearch::SQInfo::pack_code_range_, sq_info_.n_val_per_byte_,
        n_rerank_vec, n_thread_per_vec,
        item_vec_info_.vec_dim_,  item_vec_info_.max_item_n_vec_,
        vec_l_gpu_);
    CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // compute vec score
    nvtxRangePushA("rerank-computeScore");
    MatrixMultiply(gpu_resource_->handle_raft_.get_cublas_handle(), query_gpu, vec_l_gpu_,
                   query_n_vec_, probe_topk_ * item_vec_info_.max_item_n_vec_, vq_info_.vec_dim_,
                   score_table_gpu_);
    CHECK(cudaDeviceSynchronize());

    // reduce vec score
    raft::linalg::reduce(
        max_score_l_gpu_,
        score_table_gpu_,
        item_vec_info_.max_item_n_vec_,
        query_n_vec_ * probe_topk_,
        std::numeric_limits<float>::lowest(),
        true,
        true,
        gpu_resource_->handle_raft_.get_stream(),
        false,
        raft::identity_op(),
        raft::max_op(),
        raft::identity_op()
    );
    CHECK(cudaDeviceSynchronize());

    raft::linalg::reduce(
        score_l_gpu_,
        max_score_l_gpu_,
        probe_topk_,
        query_n_vec_,
        0.0f,
        true,
        false,
        gpu_resource_->handle_raft_.get_stream(),
        false,
        raft::identity_op(),
        raft::add_op(),
        raft::identity_op()
    );
    compute_time = compute_record_.get_time_second();
    nvtxRangePop();
    CHECK(cudaDeviceSynchronize());
    // thrust::transform(
    //     thrust::device,
    //     thrust::make_counting_iterator(0),
    //     thrust::make_counting_iterator((int) nprobe_topk),
    //     score_l_gpu_,
    //     [=] __device__(int candID) {
    //       float sum = 0.0f;
    //       for (int qvecID = 0; qvecID < query_n_vec_; ++qvecID) {
    //         sum += max_score_l_gpu_[candID * query_n_vec_ + qvecID];
    //       }
    //       return sum;
    //     }
    // );
  }

  void finishCompute() {
    transmit_record_.destroy();
    compute_record_.destroy();

    cudaMemFreeMarco(this->vq_code_l_gpu_);
    cudaMemFreeMarco(this->sq_code_l_gpu_);
    cudaMemFreeMarco(this->item_n_vec_l_gpu_);
    cudaMemFreeMarco(this->item_n_vec_offset_l_gpu_);
    cudaMemFreeMarco(this->vecID2vec_offset_l_gpu_);
    cudaMemFreeMarco(this->vec_l_gpu_);
    cudaMemFreeMarco(this->score_table_gpu_);
    cudaMemFreeMarco(this->max_score_l_gpu_);
    cudaMemFreeMarco(this->score_l_gpu_);
  }
};
}
#endif //RERANK_HPP
