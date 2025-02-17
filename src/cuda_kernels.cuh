#pragma once

#include <vector>

#include "markshare.hpp"

class GpuData
{
public:
    GpuData() = default;
    GpuData(const MarkShareFeas &ms_inst, const std::vector<size_t> &set1_scores, const std::vector<size_t> &set2_scores, const std::vector<size_t> &set3_scores, const std::vector<size_t> &set4_scores);

    ~GpuData();

    size_t *set1_scores{};
    size_t *set2_scores{};
    size_t *set3_scores{};
    size_t *set4_scores{};

    size_t *matrix{};
    size_t *rhs{};

    size_t m_rows{};
    size_t n_cols{};

    size_t *scores_buffer1{};
    size_t len_scores_buffer1{};

    size_t *scores_buffer2{};
    size_t len_scores_buffer2{};

    size_t *pairs_buffer1{};
    size_t len_pairs_buffer1{};

    size_t *pairs_buffer2{};
    size_t len_pairs_buffer2{};

    void init_scores_buffer(size_t n_pairs, bool first_buffer);
    void copy_pairs(const std::vector<std::pair<size_t, size_t>> &pairs, bool first_buffer);
};

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu_hashing(const GpuData &gpu_data, size_t n_q1, size_t n_q2);

void combine_scores_gpu(GpuData &gpu_data, const std::vector<std::pair<size_t, size_t>> &pairs, bool first_buffer);