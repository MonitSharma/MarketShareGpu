#pragma once

#include <vector>

#include "markshare.hpp"

struct PairsTuple
{
    PairsTuple(size_t pairs_first_, size_t pairs_second_beg_, size_t pairs_n_second_, size_t pairs_offset_)
        : pairs_first(pairs_first_), pairs_second_beg(pairs_second_beg_), pairs_n_second(pairs_n_second_), pairs_offset(pairs_offset_) {}
    size_t pairs_first;
    size_t pairs_second_beg;
    size_t pairs_n_second;
    size_t pairs_offset;
};

class GpuData
{
public:
    GpuData() = default;
    GpuData(const MarkShareFeas &ms_inst, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores);

    ~GpuData();

    int64_t n_bytes_alloc{};

    /* Required data for any computations on GPU. Is considered constant throughout the algorithm. */
    size_t *set1_scores{};
    size_t *set2_scores{};
    size_t *set3_scores{};
    size_t *set4_scores{};

    size_t *matrix{};
    size_t *rhs{};

    size_t m_rows{};
    size_t n_cols{};

    /* GPU buffers. Get resized depending on the problem. */
    __int128_t *required_buffer{};
    size_t len_required_buffer{}; /* Size of above buffers. */
    size_t n_required;

    __int128_t *search_buffer{};
    size_t len_search_buffer{};
    bool *results_search_buffer{};
    size_t len_results_buffer{};
    size_t n_search{};

    /* For new tuples approach. */
    size_t *tuples_buffer{};
    size_t len_tuples_buffer{};
    size_t n_tuples{};

    double get_gb_allocated() const
    {
        return (double)n_bytes_alloc / (1000000000);
    };

    template <typename T>
    void resize_buffer(T **buffer, size_t &buffer_size, size_t n_elems_required);

    // void resize_required(size_t n_hashes);
    // void resize_search(size_t n_hashes);

    void copy_tuples(const std::vector<PairsTuple> &tuples);
    void copy_pairs_search(const std::vector<std::pair<size_t, size_t>> &pairs);
    void copy_pairs_required(const std::vector<std::pair<size_t, size_t>> &pairs);
};

std::pair<bool, __int128_t> find_equal_hash(GpuData &gpu_data);
std::pair<size_t, size_t> find_hash_positions_gpu(GpuData &gpu_data, __int128_t hash, size_t n_p1, size_t n_p2);

void combine_and_encode_tuples_gpu(GpuData &gpu_data, const std::vector<PairsTuple> &tuples1, const std::vector<PairsTuple> &tuples2, size_t n_pairs1, size_t n_pairs2);
void combine_and_encode_gpu(GpuData &gpu_data, const std::vector<std::pair<size_t, size_t>> &pairs1, const std::vector<std::pair<size_t, size_t>> &pairs2);
void combine_and_encode_first_five_gpu(GpuData &gpu_data, const std::vector<std::pair<size_t, size_t>> &pairs1, const std::vector<std::pair<size_t, size_t>> &pairs2);
