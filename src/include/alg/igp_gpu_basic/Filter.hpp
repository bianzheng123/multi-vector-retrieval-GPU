//
// Created by Administrator on 7/8/2025.
//

#ifndef FILTER_HPP
#define FILTER_HPP

#include <raft/core/device_resources.hpp>
#include <raft/matrix/detail/select_k-inl.cuh>
#include <thrust/unique.h>
#include <thrust/binary_search.h>

#include "include/alg/igp_gpu_basic/DataStruct.hpp"

namespace VectorSetSearch
{
    __global__ void copy_ivf_n_item(const uint32_t* ivf_size_l,
                                    const uint32_t* unique_centID_l,
                                    const uint32_t n_thread, const uint32_t n_unique_centroid,
                                    uint32_t* uniqueCentID2itemID_l)
    {
        // threadID = n_thread
        const uint64_t threadID = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (threadID >= n_thread) return;
        for (uint32_t i = threadID; i < n_unique_centroid; i += n_thread)
        {
            const uint32_t centID = unique_centID_l[i];
            uniqueCentID2itemID_l[i] = ivf_size_l[centID];
        }
    }

    __global__ void add_score(const uint32_t* filter_centID_l, const float* filter_score_l,
                              const uint32_t* unique_centID_l, const uint32_t* centID_filter2unique_l,
                              const uint32_t* uniqueCentID2_n_item_l, const uint32_t* uniqueCentID2itemID_l,
                              const uint32_t max_n_ele_ivf,
                              const uint32_t query_n_vec, const uint32_t nprobe,
                              char* is_seen_l, float* item_score_l)
    {
        // threadID = query_n_vec_
        const uint64_t threadID = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (threadID >= query_n_vec) return;
        const uint32_t qvecID = threadID;

        for (uint32_t probeID = 0; probeID < nprobe; probeID++)
        {
            const uint32_t centID = filter_centID_l[qvecID * nprobe + probeID];
            const float cent_score = filter_score_l[qvecID * nprobe + probeID];
            const uint32_t unique_cent_candID = centID_filter2unique_l[qvecID * nprobe + probeID];
            assert(centID == unique_centID_l[unique_cent_candID]);

            const uint32_t ivf_n_item = uniqueCentID2_n_item_l[unique_cent_candID];
            const uint32_t* ivf_itemID_l = uniqueCentID2itemID_l + unique_cent_candID * max_n_ele_ivf;
            for(uint32_t eleID=0;eleID < ivf_n_item;eleID++) {
                const uint32_t itemID = ivf_itemID_l[eleID];
                if(is_seen_l[itemID * query_n_vec + qvecID] == false) {
                    is_seen_l[itemID * query_n_vec + qvecID] = true;
                    atomicAdd(&item_score_l[itemID], cent_score);
                }
            }
        }
    }

    class Filter
    {
    public:
        ItemVecInfo item_vec_info_;
        VQInfo vq_info_;
        IVFInfo ivf_info_;
        const GPUResource* gpu_resource_;

        uint32_t query_n_vec_, nprobe_, probe_topk_;

        TimeRecordCUDA transmit_record_, compute_record_;

        uint32_t* unique_centID_l_gpu_; // query_n_vec * nprobe, not full
        uint32_t* centID_filter2unique_l_gpu_; // query_n_vec * nprobe
        std::vector<uint32_t> unique_centID_l_; // query_n_vec * nprobe, not full
        std::vector<uint32_t> uniqueCentID2itemID_l_; // query_n_vec * nprobe * max_n_ele_ivf, padding
        uint32_t* uniqueCentID2_n_item_l_gpu_; // query_n_vec * nprobe_
        uint32_t* uniqueCentID2itemID_l_gpu_; // query_n_vec * nprobe * max_n_ele_ivf

        char* is_seen_l_gpu_{}; // n_item * query_n_vec, need reset to 0, i.e, false
        float* item_score_l_gpu_{}; // n_item, need reset to 0

        uint32_t* item_cand_indices_l_gpu_{}; // n_item, start from 0...(n_item-1)
        float* filter_score_l_gpu_{}; // probe_topk
        uint32_t* filter_itemID_l_gpu_{}; // probe_topk

        Filter() = default;

        Filter(const ItemVecInfo& item_vec_info, const VQInfo& vq_info,
               const IVFInfo& ivf_info, const GPUResource* gpu_resource,
               const uint32_t query_n_vec, const uint32_t nprobe, const uint32_t probe_topk)
        {
            this->item_vec_info_ = item_vec_info;
            this->vq_info_ = vq_info;
            this->ivf_info_ = ivf_info;
            this->gpu_resource_ = gpu_resource;

            this->query_n_vec_ = query_n_vec;
            this->nprobe_ = nprobe;
            this->probe_topk_ = probe_topk;

            CHECK(cudaMalloc(&unique_centID_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
            CHECK(cudaMalloc(&centID_filter2unique_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
            unique_centID_l_.resize(query_n_vec_ * nprobe_);
            uniqueCentID2itemID_l_.resize(query_n_vec_ * nprobe_ * ivf_info_.max_n_ele_ivf_);
            CHECK(cudaMalloc(&uniqueCentID2_n_item_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
            CHECK(cudaMalloc(&uniqueCentID2itemID_l_gpu_, query_n_vec_ * nprobe_ * ivf_info_.max_n_ele_ivf_ *
                sizeof(uint32_t)));

            CHECK(cudaMalloc(&is_seen_l_gpu_, item_vec_info_.n_item_ * query_n_vec_ * sizeof(char)));
            CHECK(cudaMalloc(&item_score_l_gpu_, item_vec_info_.n_item_ * sizeof(float)));

            CHECK(cudaMalloc(&item_cand_indices_l_gpu_, item_vec_info_.n_item_ * sizeof(uint32_t)));
            thrust::sequence(thrust::device, item_cand_indices_l_gpu_,
                             item_cand_indices_l_gpu_ + item_vec_info_.n_item_, 0);
            CHECK(cudaMalloc(&filter_score_l_gpu_, probe_topk_ * sizeof(float)));
            CHECK(cudaMalloc(&filter_itemID_l_gpu_, probe_topk_ * sizeof(uint32_t)));
        }

        void reset()
        {
            CHECK(cudaMemset(is_seen_l_gpu_, 0, item_vec_info_.n_item_ * query_n_vec_ * sizeof(char)));
            CHECK(cudaMemset(item_score_l_gpu_, 0, item_vec_info_.n_item_ * sizeof(float)));
        }

        // output: filter_itemID_l_gpu_
        void filter(const float* filter_cent_score_l_gpu, const uint32_t* filter_centID_l_gpu,
                    double& filter_transmit_time, double& filter_compute_time)
        {
            /*
             * 1. filter_centID_l -> unique_centID_l, centID_filter2unique_l, implementation: call unique function,
             * then find the index of it by the binary search
             * 2. GPU -> CPU: unique_centID_l
             * 3. CPU -> GPU: unique_itemID_l, store the array of itemID in each centroid, padding, optimize the transmit
             * 4. GPU: perform the scatter and max operation, use the shared memory to cache them
             *      output: qvec_max_score_l_gpu_, comp_item_score_l_gpu_
             */

            // compute the unique centID
            nvtxRangePushA("filter-transmit-compute_offset");
            transmit_record_.start_record();
            nvtxRangePushA("OverallTransmitTime");
            CHECK(cudaMemcpy(unique_centID_l_gpu_, filter_centID_l_gpu,
                query_n_vec_ * nprobe_ * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CHECK(cudaDeviceSynchronize());
            nvtxRangePop();
            thrust::sort(thrust::device, unique_centID_l_gpu_, unique_centID_l_gpu_ + query_n_vec_ * nprobe_);
            auto unique_end_ptr = thrust::unique(thrust::device,
                                                 unique_centID_l_gpu_, unique_centID_l_gpu_ + query_n_vec_ * nprobe_);
            const uint32_t n_unique_centroid = unique_end_ptr - unique_centID_l_gpu_;
            // transmit to cpu to get the inverted file data, at the same time compute centID_filter2unique_l_gpu_ and uniqueCentID2_n_item_l_gpu_
            thrust::lower_bound(thrust::device,
                                unique_centID_l_gpu_, unique_centID_l_gpu_ + n_unique_centroid,
                                filter_centID_l_gpu, filter_centID_l_gpu + query_n_vec_ * nprobe_,
                                centID_filter2unique_l_gpu_);

            uint64_t blockSize = 1024;
            uint32_t n_thread = n_unique_centroid;
            uint64_t gridSize = (n_thread + blockSize - 1) / blockSize;
            copy_ivf_n_item<<<gridSize, blockSize>>>(ivf_info_.ivf_size_l_gpu_, unique_centID_l_gpu_,
                                                     n_thread, n_unique_centroid,
                                                     uniqueCentID2_n_item_l_gpu_);

            nvtxRangePushA("OverallTransmitTime");
            CHECK(cudaMemcpyAsync(unique_centID_l_.data(), unique_centID_l_gpu_,
                n_unique_centroid * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (uint32_t cent_candID = 0; cent_candID < n_unique_centroid; cent_candID++)
            {
                const uint32_t centID = unique_centID_l_[cent_candID];
                const size_t ivf_offset = ivf_info_.ivf_offset_l_[centID];
                const uint32_t ivf_size = ivf_info_.ivf_size_l_[centID];
                assert(ivf_size <= ivf_info_.max_n_ele_ivf_);
                const uint32_t* ivf_itemID_start_ptr = ivf_info_.ivf_ + ivf_offset;
                uint32_t* itemID_l = uniqueCentID2itemID_l_.data() + cent_candID * ivf_info_.max_n_ele_ivf_;
                std::memcpy(itemID_l, ivf_itemID_start_ptr, ivf_size * sizeof(uint32_t));
            }
            CHECK(cudaMemcpyAsync(uniqueCentID2itemID_l_gpu_, uniqueCentID2itemID_l_.data(),
                n_unique_centroid * ivf_info_.max_n_ele_ivf_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
            filter_transmit_time = transmit_record_.get_time_second();
            CHECK(cudaDeviceSynchronize());
            nvtxRangePop();
            nvtxRangePop();

            nvtxRangePushA("filter-max_score_pair");
            compute_record_.start_record();
            blockSize = 1024;
            gridSize = (query_n_vec_ + blockSize - 1) / blockSize;
            add_score<<<gridSize, blockSize>>>(filter_centID_l_gpu, filter_cent_score_l_gpu,
                                               unique_centID_l_gpu_, centID_filter2unique_l_gpu_,
                                               uniqueCentID2_n_item_l_gpu_, uniqueCentID2itemID_l_gpu_,
                                               ivf_info_.max_n_ele_ivf_,
                                               query_n_vec_, nprobe_,
                                               is_seen_l_gpu_, item_score_l_gpu_);

            bool select_min = false;
            raft::matrix::detail::select_k(gpu_resource_->handle_raft_,
                                           item_score_l_gpu_,
                                           item_cand_indices_l_gpu_,
                                           1,
                                           item_vec_info_.n_item_,
                                           probe_topk_,
                                           filter_score_l_gpu_,
                                           filter_itemID_l_gpu_,
                                           select_min);
            filter_compute_time = compute_record_.get_time_second();
            CHECK(cudaDeviceSynchronize());
            nvtxRangePop();
        }

        void finishCompute()
        {
            transmit_record_.destroy();
            compute_record_.destroy();

            cudaMemFreeMarco(this->unique_centID_l_gpu_);
            cudaMemFreeMarco(this->centID_filter2unique_l_gpu_);
            cudaMemFreeMarco(this->uniqueCentID2_n_item_l_gpu_);
            cudaMemFreeMarco(this->uniqueCentID2itemID_l_gpu_);

            cudaMemFreeMarco(this->is_seen_l_gpu_);
            cudaMemFreeMarco(this->item_score_l_gpu_);

            cudaMemFreeMarco(this->item_cand_indices_l_gpu_);
            cudaMemFreeMarco(this->filter_score_l_gpu_);
            cudaMemFreeMarco(this->filter_itemID_l_gpu_);
        }
    };
}
#endif //FILTER_HPP
