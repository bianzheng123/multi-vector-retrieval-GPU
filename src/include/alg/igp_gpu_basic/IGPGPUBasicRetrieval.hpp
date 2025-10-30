//
// Created by Administrator on 7/8/2025.
//

#ifndef IGPGPUBASICRETRIEVAL_HPP
#define IGPGPUBASICRETRIEVAL_HPP

// Function QueryCentScore, input: ivf, query_centroid_score_l. output: query_vec_score_l, pair_ID_l
// Function Deduplicate, input: pairID_l. output: occurance_l, n_inunique_pair, use bucket sort
// Function DuplicateMemcpy, input: pairID_l, occurance_l.
// output: inunique_pair_l, inunique_size_l, inunique_offset_l
// Function Process unique_score_l, input: pairID_l, occurance_l: score_l (score_l should be atomic)
// Function Process inunique_score_l, input: inunique_pair_l, inunique_size_l, inunique_offset_l, output: score_l

/*
need to test: how to dedupliate in gpu
how to declare a atomic array
*/

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <nvtx3/nvToolsExt.h>

#include "include/util/CUDAUtil.hpp"
#include "include/alg/MatrixMulCUBLAS.hpp"
#include "include/alg/igp_gpu_basic/DataStruct.hpp"
#include "include/alg/igp_gpu_basic/GenerateVQScoreCandidate.hpp"
#include "include/alg/igp_gpu_basic/Filter.hpp"
#include "include/alg/igp_gpu_basic/Rerank.hpp"

namespace VectorSetSearch
{
    class IGPGPUBasicRetrieval
    {
    public:
        ItemVecInfo item_vec_info_;
        VQInfo vq_info_;
        IVFInfo ivf_info_;
        SQInfo sq_info_;
        std::unique_ptr<GPUResource> gpu_resource_;

        uint32_t query_n_vec_, topk_, nprobe_, probe_topk_;

        TimeRecordCUDA transmit_query_record_, gen_cand_record_, filter_record_, rerank_record_;

        float* query_gpu_; // query_n_vec * vec_dim

        GenerateVQScoreCandidate gen_cand_ins_;
        Filter filter_ins_;
        Rerank rerank_ins_;

        std::vector<float> cand_score_l_; // probe_topk
        std::vector<std::pair<float, uint32_t>> item_cand_l_; // probe_topk

        IGPGPUBasicRetrieval() = default;

        IGPGPUBasicRetrieval(const uint32_t n_item, const uint32_t max_item_n_vec,  const uint32_t min_item_n_vec, const size_t n_vec,
                       const uint32_t* item_n_vec_l, const size_t* item_n_vec_offset_l_,
                       const uint32_t n_centroid, const uint32_t vec_dim,
                       const float* centroid_l, const uint32_t* vq_code_l,
                       const size_t n_ele_ivf,
                       const size_t* ivf_offset_l, const uint32_t* ivf_size_l, const uint32_t* ivf,
                       const uint32_t n_bit, const uint32_t n_val_per_byte, const uint32_t n_packed_val_per_vec,
                       const uint32_t n_weight, const float* weight_l,
                       const uint8_t* residual_code_l)
        {
            item_vec_info_ = ItemVecInfo(n_item, max_item_n_vec, min_item_n_vec, vec_dim, n_vec,
                                         item_n_vec_l, item_n_vec_offset_l_);
            vq_info_ = VQInfo(n_centroid, vec_dim, n_vec, centroid_l, vq_code_l);
            ivf_info_ = IVFInfo(n_centroid, n_ele_ivf, ivf_offset_l, ivf_size_l, ivf);
            sq_info_ = SQInfo(n_vec, vec_dim,
                              n_bit, n_val_per_byte, n_packed_val_per_vec,
                              n_weight, weight_l, residual_code_l);
            gpu_resource_ = std::make_unique<GPUResource>();

        }

        void initSearch(const uint32_t query_n_vec, const uint32_t topk,
                        const uint32_t nprobe, const uint32_t probe_topk)
        {
            this->query_n_vec_ = query_n_vec;
            this->topk_ = topk;
            this->nprobe_ = nprobe;
            this->probe_topk_ = probe_topk;
            if (nprobe > vq_info_.n_centroid_)
            {
                spdlog::error("nprobe is larger than n_centroid, nprobe {}, n_centroid {}",
                              nprobe, vq_info_.n_centroid_);
                exit(-1);
            }


            // cublasCheckErrors(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));

            CHECK(cudaMalloc(&query_gpu_, query_n_vec_ * item_vec_info_.vec_dim_ * sizeof(float)));
            gen_cand_ins_ = GenerateVQScoreCandidate(item_vec_info_, vq_info_, gpu_resource_.get(), query_n_vec_, nprobe_);
            filter_ins_ = Filter(item_vec_info_, vq_info_, ivf_info_, gpu_resource_.get(),
                                query_n_vec_, nprobe_, probe_topk_);
            rerank_ins_ = Rerank(item_vec_info_, vq_info_, sq_info_,gpu_resource_.get(),
                                 query_n_vec_, probe_topk_);

            cand_score_l_.resize(probe_topk_);
            item_cand_l_.resize(probe_topk_);
        }

        void reset(double& reset_time)
        {
            TimeRecord record;
            record.reset();
            // another method is to initialize in cpu and copy from it
            filter_ins_.reset();
            rerank_ins_.reset();
            reset_time = record.get_elapsed_time_second();
        }

        std::vector<std::pair<float, uint32_t>> search(const float* query,
                                                       double& transmit_query_time,
                                                       double& gen_cand_time,
                                                       double& filter_time,
                                                       double& filter_transmit_time, double& filter_compute_time,
                                                       double& rerank_time,
                                                       double& rerank_transmit_time, double& rerank_compute_time,
                                                       const uint32_t queryID)
        {
            //    n_scan_vq_score = 0;
            //    n_add_vq_score = 0;
            //    n_seen_item = 0;

            nvtxRangePushA("transmit_query");
            transmit_query_record_.start_record();
            nvtxRangePushA("OverallTransmitTime");
            CHECK(cudaMemcpy(query_gpu_, query, query_n_vec_ * vq_info_.vec_dim_ * sizeof(float),
                cudaMemcpyHostToDevice));
            CHECK(cudaDeviceSynchronize());
            nvtxRangePop();
            transmit_query_time = transmit_query_record_.get_time_second();
            nvtxRangePop();

            nvtxRangePushA("generate_candidate");
            gen_cand_record_.start_record();
            gen_cand_ins_.generateCandidate(query_gpu_);
            gen_cand_time = gen_cand_record_.get_time_second();
            nvtxRangePop();

            // scatter
            // each block is a query, when finish computing the whole result, finish compute
            nvtxRangePushA("filter");
            filter_record_.start_record();
            //            spdlog::info("start filter");
            filter_ins_.filter(gen_cand_ins_.filter_cent_score_gpu_, gen_cand_ins_.filter_centID_l_gpu_,
                               filter_transmit_time, filter_compute_time);
            filter_time = filter_record_.get_time_second();
            nvtxRangePop();

            // refine
            nvtxRangePushA("rerank");
            rerank_record_.start_record();
            rerank_ins_.rerank(query_gpu_,
                               filter_ins_.filter_itemID_l_gpu_,
                               rerank_transmit_time, rerank_compute_time,
                               queryID);
            CHECK(cudaMemcpy(cand_score_l_.data(), rerank_ins_.score_l_gpu_,
                probe_topk_ * sizeof(float),
                cudaMemcpyDeviceToHost));
            CHECK(cudaDeviceSynchronize());
            rerank_time = rerank_record_.get_time_second();
            nvtxRangePop();
            for (uint32_t i = 0; i < probe_topk_; i++)
            {
                item_cand_l_[i] = std::make_pair(cand_score_l_[i], rerank_ins_.itemID_l_[i]);
            }
            std::sort(item_cand_l_.begin(), item_cand_l_.begin() + probe_topk_,
                      [](const std::pair<float, uint32_t>& l, const std::pair<float, uint32_t>& r)
                      {
                          return l.first > r.first;
                      });

            // test
            // uint32_t n_diff = 0;
            // for (uint32_t candID = 0; candID < probe_topk_; candID++)
            // {
            //     if (rerank_ins_.rerank_itemID_l_[candID] != item_cand_l_[candID].second)
            //     {
            //         n_diff++;
            //     }
            // }
            // assert(n_diff != 0);
            // end test

            return item_cand_l_;
        }

        void FinishCompute()
        {
            item_vec_info_.FinishCompute();
            vq_info_.FinishCompute();
            ivf_info_.FinishCompute();
            sq_info_.FinishCompute();
            gpu_resource_->FinishCompute();

            transmit_query_record_.destroy();
            gen_cand_record_.destroy();
            filter_record_.destroy();
            rerank_record_.destroy();

            cudaMemFreeMarco(this->query_gpu_);
            gen_cand_ins_.finishCompute();
            filter_ins_.finishCompute();
            rerank_ins_.finishCompute();
        }
    };
}
#endif //IGPGPUBASICRETRIEVAL_HPP
