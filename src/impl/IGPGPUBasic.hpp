//
// Created by Administrator on 7/8/2025.
//

#ifndef IGPGPUBASIC_HPP
#define IGPGPUBASIC_HPP

#include <spdlog/spdlog.h>
#include <fstream>
#include <cnpy.h>

#include "include/struct/IVFIndex.hpp"
#include "include/alg/refine/ResidualScalarQuantizationCPP.hpp"
#include "include/util/TimeMemory.hpp"
#include "include/alg/igp_gpu_basic/IGPGPUBasicRetrieval.hpp"

namespace VectorSetSearch::Method {
struct IGPGPUBasicResult {
  std::vector<float> result_score_l; // n_query * topk
  std::vector<uint32_t> result_ID_l; // n_query * topk
  std::vector<double> compute_time_l; // n_query

  std::vector<double> reset_time_l; // n_query
  std::vector<double> transmit_query_time_l; // n_query
  std::vector<double> cand_gen_time_l; // n_query

  std::vector<double> filter_time_l; // n_query
  std::vector<double> filter_transmit_time_l; // n_query
  std::vector<double> filter_compute_time_l; // n_query

  std::vector<double> rerank_time_l; // n_query
  std::vector<double> rerank_transmit_time_l; // n_query
  std::vector<double> rerank_compute_time_l; // n_query

  uint32_t n_query, topk;
};

class IGPGPUBasic {
 public:
  cnpy::NpyArray item_n_vec_l_npy_;
  uint32_t _n_item{}, max_item_n_vec_{}, min_item_n_vec_{};
  size_t _n_vecs{};
  const uint32_t *_item_n_vecs_l; // n_item
  std::vector<size_t> _item_n_vecs_offset_l; // n_item

  cnpy::NpyArray centroid_l_npy_;
  uint32_t _n_centroid{}, _vec_dim{};
  const float *_centroid_l; // _n_centroid * vec_dim
  cnpy::NpyArray vq_code_l_npy_;
  const uint32_t *_code_l; // n_vecs

  std::vector<std::vector<uint32_t>> _centroid2itemID_l; // n_centroid
  size_t n_ele_ivf_;
  std::vector<size_t> ivf_offset_l_; // n_centroid
  std::vector<uint32_t> ivf_size_l_; // n_centroid
  std::vector<uint32_t> ivf_; // n_ele_ivf_

  uint32_t n_bit_, n_val_per_byte_, n_packed_val_per_vec_;

  cnpy::NpyArray weight_l_npy_;
  uint32_t n_weight_;
  const float *weight_l_; // n_weight

  cnpy::NpyArray residual_code_npy_;
  const uint8_t *residual_code_l_; // n_vecs * n_packed_val_per_vec_

  IGPGPUBasicRetrieval retrieval_ins_;

  IGPGPUBasic() = default;

  IGPGPUBasic(const std::string &item_n_vec_l_fname) {
    item_n_vec_l_npy_ = cnpy::npy_load(item_n_vec_l_fname);
    assert(item_n_vec_l_npy_.word_size == sizeof(uint32_t));
    assert(item_n_vec_l_npy_.shape.size() == 1);
    const uint32_t n_item = item_n_vec_l_npy_.shape[0];
    const uint32_t *item_n_vec_l = item_n_vec_l_npy_.data<uint32_t>();

    _n_item = n_item;
    auto max_ptr = std::max_element(item_n_vec_l, item_n_vec_l + n_item);
    max_item_n_vec_ = *max_ptr;
    auto min_ptr = std::min_element(item_n_vec_l, item_n_vec_l + n_item);
    min_item_n_vec_ = *min_ptr;

    _item_n_vecs_l = item_n_vec_l_npy_.data<uint32_t>();
    _item_n_vecs_offset_l.resize(_n_item);

    size_t n_vecs = _item_n_vecs_l[0];
    _item_n_vecs_offset_l[0] = 0;
    for (uint32_t itemID = 1; itemID < _n_item; itemID++) {
      n_vecs += _item_n_vecs_l[itemID];
      _item_n_vecs_offset_l[itemID] = _item_n_vecs_offset_l[itemID - 1] + _item_n_vecs_l[itemID - 1];
    }
    _n_vecs = n_vecs;
  }

  void buildIVFIndex() {
    ivf_offset_l_.resize(_n_centroid);
    ivf_offset_l_[0] = 0;
    for (uint32_t centID = 1; centID < _n_centroid; centID++) {
      ivf_offset_l_[centID] = ivf_offset_l_[centID - 1] + _centroid2itemID_l[centID - 1].size();
    }

    ivf_size_l_.resize(_n_centroid);
    for (uint32_t centID = 0; centID < _n_centroid; centID++) {
      ivf_size_l_[centID] = _centroid2itemID_l[centID].size();
    }

    size_t n_ele_ivf = 0;
    for (uint32_t centID = 0; centID < _n_centroid; centID++) {
      n_ele_ivf += _centroid2itemID_l[centID].size();
    }
    ivf_.resize(n_ele_ivf);
    this->n_ele_ivf_ = n_ele_ivf;

    for (uint32_t centID = 0; centID < _n_centroid; centID++) {
      const size_t offset = ivf_offset_l_[centID];
      const uint32_t n_ele = _centroid2itemID_l[centID].size();
      std::memcpy(ivf_.data() + offset, _centroid2itemID_l[centID].data(), n_ele * sizeof(uint32_t));
    }
  }

  bool buildIndex(const std::string &centroid_l_fname, const std::string &vq_code_l_fname,
                  const std::string &ivf_index_fname,
                  const std::string &weight_l_fname, const std::string &residual_code_l_fname,
                  const uint32_t n_centroid, const uint32_t n_bit) {
    centroid_l_npy_ = cnpy::npy_load(centroid_l_fname);
    assert(centroid_l_npy_.word_size == sizeof(float));
    assert(centroid_l_npy_.shape.size() == 2);
    _n_centroid = centroid_l_npy_.shape[0];
    _vec_dim = centroid_l_npy_.shape[1];
    _centroid_l = centroid_l_npy_.data<float>();
    assert(_n_centroid == n_centroid);

    vq_code_l_npy_ = cnpy::npy_load(vq_code_l_fname);
    assert(vq_code_l_npy_.word_size == sizeof(uint32_t));
    assert(vq_code_l_npy_.shape.size() == 1);
    assert(vq_code_l_npy_.shape[0] == _n_vecs);
    _code_l = vq_code_l_npy_.data<uint32_t>();

#ifndef NDEBUG
    for (uint32_t vecID = 0; vecID < _n_vecs; vecID++) {
      assert(_code_l[vecID] < _n_centroid);
    }
#endif

    _centroid2itemID_l = Method::readIVFIndex(ivf_index_fname);

#ifndef NDEBUG
    assert(_centroid2itemID_l.size() == _n_centroid);
    size_t n_element = 0;
    for (size_t centID = 0; centID < _n_centroid; centID++) {
      n_element += _centroid2itemID_l[centID].size();
    }
    assert(n_element >= _n_item);
#endif
    // _centroid2itemID_l -> n_ele_ivf_, ivf_offset_l_, ivf_size_l_, ivf_
    buildIVFIndex();

    n_bit_ = n_bit;
    constexpr uint32_t n_bit_per_byte = 8;
    n_val_per_byte_ = n_bit_per_byte / n_bit_;
    assert(n_bit_per_byte % n_bit_ == 0);
    n_packed_val_per_vec_ = (_vec_dim + n_val_per_byte_ - 1) / n_val_per_byte_;

    weight_l_npy_ = cnpy::npy_load(weight_l_fname);
    assert(weight_l_npy_.word_size == sizeof(float));
    assert(weight_l_npy_.shape.size() == 1);
    n_weight_ = weight_l_npy_.shape[0];
    weight_l_ = weight_l_npy_.data<float>();

    residual_code_npy_ = cnpy::npy_load(residual_code_l_fname);
    assert(residual_code_npy_.word_size == sizeof(uint8_t));
    assert(residual_code_npy_.shape.size() == 1);
    assert(residual_code_npy_.shape[0] == _n_vecs * n_packed_val_per_vec_);
    residual_code_l_ = residual_code_npy_.data<uint8_t>();

    retrieval_ins_ = IGPGPUBasicRetrieval(_n_item, max_item_n_vec_, min_item_n_vec_, _n_vecs,
                                    _item_n_vecs_l, _item_n_vecs_offset_l.data(),
                                    _n_centroid, _vec_dim,
                                    _centroid_l, _code_l,
                                    n_ele_ivf_,
                                    ivf_offset_l_.data(), ivf_size_l_.data(), ivf_.data(),
                                    n_bit_, n_val_per_byte_, n_packed_val_per_vec_,
                                    n_weight_, weight_l_,
                                    residual_code_l_);

    return true;
  }

  IGPGPUBasicResult search(const float *query_l, const uint32_t n_query, const uint32_t query_n_vec,
                     const uint32_t topk,
                     const uint32_t nprobe, const uint32_t probe_topk) {
    if (probe_topk < topk) {
      spdlog::error("the number of refined topk is smaller than the number of returned topk, program exit");
      exit(-1);
    }
    //            if (!(refine_topk <= nprobe && nprobe <= _n_centroid)) {
    //                spdlog::error(
    //                        "nprobe is either smaller than refine_topk or larger than n_centroid, program exit");
    //                exit(-1);
    //            }


    retrieval_ins_.initSearch(query_n_vec, topk, nprobe, probe_topk);

    std::vector<float> result_score_l((int64_t) n_query * topk);
    std::vector<uint32_t> result_ID_l((int64_t) n_query * topk);
    std::vector<double> compute_time_l(n_query);

    std::vector<double> reset_time_l(n_query);
    std::vector<double> transmit_query_time_l(n_query);
    std::vector<double> cand_gen_time_l(n_query);
    std::vector<double> filter_time_l(n_query);
    std::vector<double> filter_transmit_time_l(n_query);
    std::vector<double> filter_compute_time_l(n_query);
    std::vector<double> rerank_time_l(n_query);
    std::vector<double> rerank_transmit_time_l(n_query);
    std::vector<double> rerank_compute_time_l(n_query);

    for(uint32_t queryID = 0; queryID < std::min(n_query, 10u); queryID++) {
      const float *query = query_l + queryID * query_n_vec * _vec_dim;
      double reset_time;
      double transmit_query_time, cand_gen_time;
      double filter_time, filter_transmit_time, filter_compute_time;
      double rerank_time, rerank_transmit_time, rerank_compute_time;
      retrieval_ins_.reset(reset_time);
      std::vector<std::pair<float, uint32_t>> res = retrieval_ins_.search(query,
                                                                          transmit_query_time,
                                                                          cand_gen_time,
                                                                          filter_time,
                                                                          filter_transmit_time,
                                                                          filter_compute_time,
                                                                          rerank_time,
                                                                          rerank_transmit_time,
                                                                          rerank_compute_time, -1);
    }
    spdlog::info("finish warming");

    TimeRecord record;
    for (uint32_t queryID = 0; queryID < n_query; queryID++) {
      if (queryID % 100 == 0) {
        spdlog::info("start processing queryID {}", queryID);
      }
//      spdlog::info("queryID {}", queryID);
      nvtxRangePushA((std::string("queryID ") + std::to_string(queryID)).c_str());
      record.reset();
      nvtxRangePushA("Reset");
      double reset_time;
      retrieval_ins_.reset(reset_time);
      nvtxRangePop();

      const float *query = query_l + queryID * query_n_vec * _vec_dim;
      double transmit_query_time, cand_gen_time;
      double filter_time, filter_transmit_time, filter_compute_time;
      double rerank_time, rerank_transmit_time, rerank_compute_time;

      std::vector<std::pair<float, uint32_t>> res = retrieval_ins_.search(query,
                                                                          transmit_query_time,
                                                                          cand_gen_time,
                                                                          filter_time,
                                                                          filter_transmit_time,
                                                                          filter_compute_time,
                                                                          rerank_time,
                                                                          rerank_transmit_time,
                                                                          rerank_compute_time,
                                                                          queryID);

      for (uint32_t candID = 0; candID < topk; candID++) {
        const int64_t insert_offset = (int64_t) queryID * topk;
        result_score_l[insert_offset + (int64_t) candID] = res[candID].first;
        result_ID_l[insert_offset + (int64_t) candID] = res[candID].second;
      }
      nvtxRangePop();

      compute_time_l[queryID] = record.get_elapsed_time_second();

      reset_time_l[queryID] = reset_time;
      transmit_query_time_l[queryID] = transmit_query_time;
      cand_gen_time_l[queryID] = cand_gen_time;
      filter_time_l[queryID] = filter_time;
      filter_transmit_time_l[queryID] = filter_transmit_time;
      filter_compute_time_l[queryID] = filter_compute_time;
      rerank_time_l[queryID] = rerank_time;
      rerank_transmit_time_l[queryID] = rerank_transmit_time;
      rerank_compute_time_l[queryID] = rerank_compute_time;
    }
    retrieval_ins_.FinishCompute();

    IGPGPUBasicResult result;
    result.result_score_l = result_score_l;
    result.result_ID_l = result_ID_l;
    result.compute_time_l = compute_time_l;

    result.reset_time_l = reset_time_l;
    result.transmit_query_time_l = transmit_query_time_l;
    result.cand_gen_time_l = cand_gen_time_l;

    result.filter_time_l = filter_time_l;
    result.filter_transmit_time_l = filter_transmit_time_l;
    result.filter_compute_time_l = filter_compute_time_l;

    result.rerank_time_l = rerank_time_l;
    result.rerank_transmit_time_l = rerank_transmit_time_l;
    result.rerank_compute_time_l = rerank_compute_time_l;

    result.n_query = n_query;
    result.topk = topk;

    return result;
  }
};
}
#endif //IGPGPUBASIC_HPP
