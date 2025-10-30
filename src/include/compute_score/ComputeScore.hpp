//
// Created by bianzheng on 2023/7/15.
//

#ifndef VECTORSETSEARCH_COMPUTESCORE_HPP
#define VECTORSETSEARCH_COMPUTESCORE_HPP

#include <vector>
#include <parallel/algorithm>
#include <thread>
#include <spdlog/spdlog.h>

#include "include/compute_score/GPUComputeScore.hpp"


namespace VectorSetSearch {
    class ComputeScore {

        GPUComputeScore gpu_;
    public:

        ComputeScore() = default;

        inline ComputeScore(const float *query_vecs_l,
                            const uint32_t &n_query, const uint32_t &query_n_vecs, const uint32_t &vec_dim) {

            spdlog::info("use GPU");
            gpu_ = GPUComputeScore(query_vecs_l, n_query, query_n_vecs, vec_dim);

        }

        void computeItemScore(const float** item_vecs_l,
                              const uint32_t* item_n_vecs_l, const uint32_t &n_item,
                              float *const distance_l) {

            gpu_.computeItemScore(item_vecs_l, item_n_vecs_l, n_item, distance_l);
        }

        void finishCompute() {
            gpu_.finishCompute();
        }
    };

}
#endif //VECTORSETSEARCH_COMPUTESCORE_HPP
