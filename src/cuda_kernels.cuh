#pragma once

#include <vector>

#include "markshare.hpp"

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
    size_t *required_sort_sequence_buffer{};
    size_t len_required_buffer{}; /* Size of above buffers. */
    size_t n_required;

    __int128_t *search_buffer{};
    size_t *results_search_buffer{};
    size_t len_search_buffer{}; /* Size of above buffers. */
    size_t n_search;

    double get_gb_allocated() const
    {
        return (double)n_bytes_alloc / (1000000000);
    };

    void copy_pairs_search(const std::vector<std::pair<size_t, size_t>> &pairs);
    void copy_pairs_required(const std::vector<std::pair<size_t, size_t>> &pairs);
};

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu_hashing(GpuData &gpu_data, size_t n_p1, size_t n_p2);

void combine_and_encode_gpu(GpuData &gpu_data, const std::vector<std::pair<size_t, size_t>> &pairs1, const std::vector<std::pair<size_t, size_t>> &pairs2);