#pragma once

#include <vector>

#include "markshare.hpp"

class GpuData
{
public:

    GpuData() = default;
    GpuData(const MarkShareFeas &ms_inst, const std::vector<std::vector<size_t>>& set1_subsets, const std::vector<std::vector<size_t>>& set2_subsets, const std::vector<std::vector<size_t>>& set3_subsets, const std::vector<std::vector<size_t>>& set4_subsets, const std::vector<size_t>& asc_indicies_set2_weights, const std::vector<size_t>& desc_indicies_set4_weights);

    ~GpuData();

    size_t* set1_subsets;
    size_t* set1_subsets_beg;

    size_t* set2_subsets;
    size_t* set2_subsets_beg;

    size_t* asc_indicies_set2_weights;

    size_t* set3_subsets;
    size_t* set3_subsets_beg;

    size_t* set4_subsets;
    size_t* set4_subsets_beg;

    size_t* desc_indicies_set4_weights;

    size_t* matrix;
    size_t* rhs;

    size_t m_rows;
    size_t n_cols;
};

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu(const GpuData& gpu_data, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2);

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu_hashing(const GpuData& gpu_data, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2);

void compute_scores_gpu(const GpuData& gpu_data, std::vector<size_t>& scores, const std::vector<std::pair<size_t, size_t>>& pairs, const std::vector<std::vector<size_t>>& pair_first_subsets,
    const std::vector<std::vector<size_t>>& pair_second_subsets, const std::vector<size_t> pair_second_map, const std::vector<size_t> offsets);