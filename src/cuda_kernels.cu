#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "markshare.hpp"

__global__ void check_sums(const size_t *val1, const size_t *val2, const size_t *rhs, size_t *solution, size_t n_val1, size_t n_val2, size_t m_rhs, )
{
    int i1 = blockIdx.x * blockDim.x + threadIdx.x; // Thread for index i1
    int i2 = blockIdx.y * blockDim.y + threadIdx.y; // Thread for index i2

    if (i1 < n_val1 && i2 < n_val2)
    {
        for (int j = 1; j < m_rhs; ++j)
        {
            const size_t sum = val1[i1 * m + j] + val2[i2 * m + j];
            if (sum != rhs[j])
                break;
        }
        *solution = i1 * n_val2 + i2;
        return;
    }
}

std::pair<bool, std::pair<size_t, size_t>> evaluate_solutions_gpu(const MarkShareFeas &ms_inst, const std::vector<bool> &feas_q1, const std::vector<bool> &feas_q2, const std::vector<size_t> &scores_q1, const std::vector<size_t> &scores_q2, size_t n_q1, size_t n_q2)
{
    size_t *d_scores_q1, *d_scores_q2, *d_rhs, *d_solution;
    size_t m = ms_inst.m();

    cudaMalloc(&d_scores_q1, scores_q1.size() * sizeof(size_t));
    cudaMalloc(&d_scores_q2, scores_q2.size() * sizeof(size_t));
    cudaMalloc(&d_rhs, m * sizeof(size_t));
    cudaMalloc(&d_solution, sizeof(size_t));

    size_t sol_invalid = scores_q1.size() * scores_q1.size() + 1;
    size_t result;
    cudaMemcpy(d_scores_q1, scores_q1.data(), scores_q1.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores_q2, scores_q2.data(), scores_q2.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, h_rhs.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_solution, &sol_invalid, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockDim(16, 16);                                                                // Threads per block
    dim3 gridDim((n1 + blockDim.x - 1) / blockDim.x, (n2 + blockDim.y - 1) / blockDim.y); // Blocks per grid

    // Launch kernel
    check_sums<<<gridDim, blockDim>>>(d_scores_q1, d_scores_q2, d_rhs, d_solution, n_q1, n_q2, m);

    // Copy result back to host
    cudaMemcpy(&result, d_solution, sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(d_scores_q1);
    cudaFree(d_scores_q2);
    cudaFree(d_rhs);
    cudaFree(d_solution);

    if (result != sol_invalid)
    {
        printf("GPU found solution!\n");
        size_t i_q1 = result / n_q2;
        size_t i_q2 = result % n_q2;

        return {true, {i_q1, i_q2}};
    }
    else
    {
        return {false, {n_q1, n_q2}};
    }
}