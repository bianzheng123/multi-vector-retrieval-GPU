//
// Created by Administrator on 7/8/2025.
//

#ifndef GENERATEVQSCORECANDIDATE_HPP
#define GENERATEVQSCORECANDIDATE_HPP

#include <raft/matrix/detail/select_k-inl.cuh>

#include <cstdint>
#include "include/alg/igp_gpu_basic/DataStruct.hpp"

namespace VectorSetSearch
{
    class GenerateVQScoreCandidate
    {
    public:
        ItemVecInfo item_vec_info_;
        VQInfo vq_info_;
        const GPUResource* gpu_resource_;

        uint32_t query_n_vec_, nprobe_;

        float* vq_score_table_gpu_; // query_n_vec * n_centroid
        float* filter_cent_score_gpu_; // query_n_vec * nprobe_
        uint32_t* filter_centID_l_gpu_; // query_n_vec * nprobe_

        GenerateVQScoreCandidate() = default;

        GenerateVQScoreCandidate(const ItemVecInfo& item_vec_info, const VQInfo& vq_info,
                                const GPUResource* gpu_resource,
                                 const uint32_t query_n_vec, const uint32_t nprobe)
        {
            spdlog::info("use GenerateVQScoreCandidate Old GPU version, matrix multiply");
            this->item_vec_info_ = item_vec_info;
            this->vq_info_ = vq_info;
            this->gpu_resource_ = gpu_resource;

            this->query_n_vec_ = query_n_vec;
            this->nprobe_ = nprobe;

            CHECK(cudaMalloc(&vq_score_table_gpu_, query_n_vec_ * vq_info_.n_centroid_ * sizeof(float)));
            CHECK(cudaMalloc(&filter_cent_score_gpu_, query_n_vec_ * nprobe_ * sizeof(float)));
            CHECK(cudaMalloc(&filter_centID_l_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t)));
        }

        // void matrixMultiplyTest(const float* vec1_l, const float* vec2_l,
        //                         int n_vec1, int n_vec2, int vec_dim, float* result)
        // {
        //     for (int vecID1 = 0; vecID1 < n_vec1; ++vecID1)
        //     {
        //         for (int vecID2 = 0; vecID2 < n_vec2; ++vecID2)
        //         {
        //             float sum = 0.0f;
        //             for (int dim = 0; dim < vec_dim; ++dim)
        //             {
        //                 sum += vec1_l[vecID1 * vec_dim + dim] * vec2_l[vecID2 * vec_dim + dim];
        //             }
        //             result[vecID1 * n_vec2 + vecID2] = sum;
        //         }
        //     }
        // }

        void generateCandidate(const float* query_gpu)
        {
            // compute and sort the score matrix
            MatrixMultiply(gpu_resource_->handle_, query_gpu, vq_info_.centroid_l_gpu_,
                           query_n_vec_, vq_info_.n_centroid_, vq_info_.vec_dim_,
                           vq_score_table_gpu_);
            CHECK(cudaDeviceSynchronize());

            thrust::for_each(thrust::device, vq_score_table_gpu_,
                             vq_score_table_gpu_ + query_n_vec_ * vq_info_.n_centroid_,
                             [] __device__(float& x) { x += 1.0f; });
            CHECK(cudaDeviceSynchronize());

            // test
            // thrust::for_each(
            //     thrust::device,
            //     thrust::make_zip_iterator(thrust::make_tuple(vq_score_table_sort_gpu_, vq_score_table_gpu_)),
            //     thrust::make_zip_iterator(thrust::make_tuple(
            //         vq_score_table_sort_gpu_ + query_n_vec_ * vq_info_.n_centroid_,
            //         vq_score_table_gpu_ + query_n_vec_ * vq_info_.n_centroid_)),
            //     [=]__device__(auto& tup)
            //     {
            //         assert(thrust::get<0>(tup) == thrust::get<1>(tup));
            //     }
            // );
            // std::vector<float> vq_score_table_before(query_n_vec_ * vq_info_.n_centroid_);
            // CHECK(cudaMemcpy(vq_score_table_before.data(), vq_score_table_gpu_,
            //     query_n_vec_ * vq_info_.n_centroid_ * sizeof(float), cudaMemcpyDeviceToHost));
            // CHECK(cudaDeviceSynchronize());
            // end test

            const uint32_t* input_idx_l = nullptr;
            bool select_min = false;
            raft::matrix::detail::select_k(gpu_resource_->handle_raft_,
                                           vq_score_table_gpu_,
                                           input_idx_l,
                                           query_n_vec_,
                                           vq_info_.n_centroid_,
                                           nprobe_,
                                           filter_cent_score_gpu_,
                                           filter_centID_l_gpu_,
                                           select_min);

            CHECK(cudaDeviceSynchronize());

            // test
            // std::vector<float> vq_score_table(query_n_vec_ * vq_info_.n_centroid_);
            // matrixMultiplyTest(query_cpu, vq_info_.centroid_l_,
            //                    (int)query_n_vec_, (int)vq_info_.n_centroid_, (int)vq_info_.vec_dim_,
            //                    vq_score_table.data());
            // for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
            // {
            //     for (uint32_t centID = 0; centID < vq_info_.n_centroid_; centID++)
            //     {
            //         const uint32_t eleID = qvecID * vq_info_.n_centroid_ + centID;
            //         vq_score_table[eleID] = vq_score_table[eleID] + 1.0f;
            //     }
            // }
            //
            // std::vector<float> vq_score_table_from_gpu(query_n_vec_ * vq_info_.n_centroid_);
            // CHECK(cudaMemcpy(vq_score_table_from_gpu.data(), vq_score_table_gpu_,
            //     query_n_vec_ * vq_info_.n_centroid_ * sizeof(float), cudaMemcpyDeviceToHost));
            // CHECK(cudaDeviceSynchronize());
            //
            // for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
            // {
            //     for (uint32_t centID = 0; centID < vq_info_.n_centroid_; centID++)
            //     {
            //         const uint32_t eleID = qvecID * vq_info_.n_centroid_ + centID;
            //         if (!(fabs(vq_score_table[eleID] - vq_score_table_from_gpu[eleID]) < 1e-3))
            //         {
            //             spdlog::error("eleID {}, vq_score_table {:.4f}, vq_score_table_from_gpu {:.4f}",
            //                           eleID, vq_score_table[eleID], vq_score_table_from_gpu[eleID]);
            //         }
            //         assert(fabs(vq_score_table[eleID] - vq_score_table_from_gpu[eleID]) < 1e-3);
            //     }
            // }
            //
            // for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
            // {
            //     for (uint32_t centID = 0; centID < vq_info_.n_centroid_; centID++)
            //     {
            //         const uint32_t eleID = qvecID * vq_info_.n_centroid_ + centID;
            //         if (!(fabs(vq_score_table_before[eleID] - vq_score_table[eleID]) < 1e-3))
            //         {
            //             spdlog::error("eleID {}, vq_score_table_before[eleID] {:.4f}, vq_score_table[centID] {:.4f}",
            //                           eleID, vq_score_table_before[eleID], vq_score_table[centID]);
            //         }
            //         assert(fabs(vq_score_table_before[eleID] - vq_score_table[eleID]) < 1e-3);
            //     }
            // }
            // spdlog::info("VQ score table verified");
            // end test

//            CHECK(cudaMemcpy(filter_centID_l_gpu_, vq_score_table_idx_gpu_, query_n_vec_ * nprobe_ * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
//            for (uint32_t qvecID = 0; qvecID < query_n_vec_; qvecID++)
//            {
//                CHECK(cudaMemcpyAsync(filter_centID_l_gpu_ + qvecID * nprobe_,
////                    vq_score_table_idx_gpu_ + qvecID * vq_info_.n_centroid_,
//                    vq_score_table_idx_gpu_ + qvecID * nprobe_,
//                    nprobe_ * sizeof(uint32_t),
//                    cudaMemcpyDeviceToDevice));
//            }
//            CHECK(cudaDeviceSynchronize());
        }

        void finishCompute()
        {
            cudaMemFreeMarco(this->vq_score_table_gpu_);
            cudaMemFreeMarco(this->filter_cent_score_gpu_);
            cudaMemFreeMarco(this->filter_centID_l_gpu_);
        }
    };
}
#endif //GENERATEVQSCORECANDIDATE_HPP
