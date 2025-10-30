//
// Created by Administrator on 2025/5/19.
//

#ifndef IVFINDEX_HPP
#define IVFINDEX_HPP

#include <spdlog/spdlog.h>
#include <iostream>
#include <set>
#include <fstream>

namespace VectorSetSearch::Method
{
    template <typename T>
    void writeBinaryDataIVF(std::ostream& out, const T& podRef)
    {
        out.write((char*)&podRef, sizeof(T));
    }

    template <typename T>
    void readBinaryDataIVF(std::istream& in, T& podRef)
    {
        in.read((char*)&podRef, sizeof(T));
    }

    inline std::vector<std::vector<uint32_t>> readIVFIndex(const std::string& location)
    {
        std::cout << "Loading inverted index from " << location << std::endl;
        std::ifstream input(location, std::ios::binary);
        std::streampos position;

        uint32_t n_centroid;
        size_t n_ele_ivf;

        readBinaryDataIVF(input, n_centroid);
        readBinaryDataIVF(input, n_ele_ivf);
        spdlog::info("n_centroid {}, n_ele_ivf {}", n_centroid, n_ele_ivf);

        std::vector<size_t> ivf_offset_l(n_centroid);
        input.read((char*)ivf_offset_l.data(), n_centroid * sizeof(size_t));

        std::vector<uint32_t> ivf_size_l(n_centroid);
        input.read((char*)ivf_size_l.data(), n_centroid * sizeof(uint32_t));

        std::vector<uint32_t> ivf(n_ele_ivf);
        input.read((char*)ivf.data(), n_ele_ivf * sizeof(uint32_t));

        std::vector<std::vector<uint32_t>> centroid2itemID_l(n_centroid);
        for (uint32_t centID = 0; centID < n_centroid; centID++)
        {
            uint32_t offset = ivf_offset_l[centID];
            uint32_t size = ivf_size_l[centID];
            centroid2itemID_l[centID].resize(size);
            std::memcpy(centroid2itemID_l[centID].data(), ivf.data() + offset, size * sizeof(uint32_t));
        }

        return centroid2itemID_l;
    }

    class IVFIndex
    {
    public:
        size_t n_vecs_{};
        uint32_t n_item_{};
        std::vector<uint32_t> item_n_vecs_l_; // n_item
        std::vector<size_t> item_n_vecs_offset_l_; // n_item

        uint32_t n_centroid_{};
        std::vector<uint32_t> code_l_; // n_vecs
        std::vector<std::vector<uint32_t>> centroid2itemID_l_; // n_centroid
        std::vector<size_t> ivf_offset_l_; // shape: n_centroid
        std::vector<uint32_t> ivf_size_l_; // shape: n_centroid
        std::vector<uint32_t> ivf_; // shape: sum(ivf_offset_l)

        IVFIndex() = default;

        IVFIndex(const std::vector<uint32_t>& vq_code_l,
                 const std::vector<uint32_t>& item_n_vec_l,
                 const size_t n_vec, const uint32_t n_item, const uint32_t n_centroid)
        {
            this->n_vecs_ = n_vec;
            this->n_item_ = n_item;
            this->item_n_vecs_l_ = item_n_vec_l;

            this->n_centroid_ = n_centroid;
            this->code_l_ = vq_code_l;
            assert(code_l_.size() == n_vecs_);
            assert(item_n_vecs_l_.size() == n_item_);
        }

        void build()
        {
            item_n_vecs_offset_l_.resize(n_item_);
            size_t n_vecs = item_n_vecs_l_[0];
            item_n_vecs_offset_l_[0] = 0;
            for (uint32_t itemID = 1; itemID < n_item_; itemID++)
            {
                n_vecs += item_n_vecs_l_[itemID];
                item_n_vecs_offset_l_[itemID] = item_n_vecs_offset_l_[itemID - 1] + item_n_vecs_l_[itemID - 1];
            }

            centroid2itemID_l_.resize(n_centroid_);
#ifndef NDEBUG
            for (uint32_t vecID = 0; vecID < n_vecs_; vecID++)
            {
                assert(code_l_[vecID] < n_centroid_);
            }
#endif
            assert(centroid2itemID_l_.size() == n_centroid_);

            for (uint32_t itemID = 0; itemID < n_item_; itemID++)
            {
                const size_t start_vecID = item_n_vecs_offset_l_[itemID];
                const uint32_t item_n_vecs = item_n_vecs_l_[itemID];
                std::set<uint32_t> centroidID_s;
                for (size_t item_vecID = 0; item_vecID < item_n_vecs; item_vecID++)
                {
                    const size_t vecID = item_vecID + start_vecID;
                    const uint32_t centroidID = code_l_[vecID];
                    centroidID_s.insert(centroidID);
                }

                for (size_t centroidID : centroidID_s)
                {
                    centroid2itemID_l_[centroidID].push_back(itemID);
                }
            }

#ifndef NDEBUG
            assert(centroid2itemID_l_.size() == n_centroid_);
            size_t n_element = 0;
            for (size_t centID = 0; centID < n_centroid_; centID++)
            {
                n_element += centroid2itemID_l_[centID].size();
            }
            assert(n_element >= n_item_);
#endif

            ivf_offset_l_.resize(n_centroid_);
            ivf_offset_l_[0] = 0;
            for (uint32_t centID = 1; centID < n_centroid_; centID++)
            {
                ivf_offset_l_[centID] = ivf_offset_l_[centID - 1] + centroid2itemID_l_[centID - 1].size();
            }

            ivf_size_l_.resize(n_centroid_);
            for (uint32_t centID = 0; centID < n_centroid_; centID++)
            {
                ivf_size_l_[centID] = centroid2itemID_l_[centID].size();
            }

            size_t n_ele_ivf = 0;
            for (uint32_t centID = 0; centID < n_centroid_; centID++)
            {
                n_ele_ivf += centroid2itemID_l_[centID].size();
            }
            ivf_.resize(n_ele_ivf);

            for (uint32_t centID = 0; centID < n_centroid_; centID++)
            {
                const size_t offset = ivf_offset_l_[centID];
                const uint32_t n_ele = centroid2itemID_l_[centID].size();
                std::memcpy(ivf_.data() + offset, centroid2itemID_l_[centID].data(), n_ele * sizeof(uint32_t));
            }
        }

        void save(const std::string& location)
        {
            std::cout << "Saving index to " << location.c_str() << "\n";
            std::ofstream output(location, std::ios::binary | std::ios::out);
            std::streampos position;

            const size_t n_ele_ivf = ivf_.size();
            writeBinaryDataIVF(output, n_centroid_);
            writeBinaryDataIVF(output, n_ele_ivf);

            output.write((char*)ivf_offset_l_.data(), n_centroid_ * sizeof(size_t));
            output.write((char*)ivf_size_l_.data(), n_centroid_ * sizeof(uint32_t));
            output.write((char*)ivf_.data(), n_ele_ivf * sizeof(uint32_t));

            output.close();
        };
    };
}
#endif //IVFINDEX_HPP
