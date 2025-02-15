#pragma once

#include <vector>

#include "markshare.hpp"

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu(const MarkShareFeas &ms_inst, const std::vector<bool> &feas_q1, const std::vector<bool> &feas_q2, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2);